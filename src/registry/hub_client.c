#include "registry/registry_internal.h"
#include "core/log.h"

#include <curl/curl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <yyjson/yyjson.h>

static char *dup_str(const char *s) {
    if (!s)
        return NULL;
    size_t n = strlen(s);
    char  *p = malloc(n + 1);
    if (p)
        memcpy(p, s, n + 1);
    return p;
}

int reg_parse_file_plan(const char *json, size_t len, char ***files, size_t *n) {
    *files = NULL;
    *n = 0;

    yyjson_doc *doc = yyjson_read(json, len, 0);
    if (!doc)
        return -1;

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *siblings = yyjson_obj_get(root, "siblings");
    if (!yyjson_is_arr(siblings)) {
        yyjson_doc_free(doc);
        return -1;
    }

    size_t arr_len = yyjson_arr_size(siblings);
    if (arr_len == 0) {
        yyjson_doc_free(doc);
        return 0;
    }

    char **result = calloc(arr_len, sizeof(char *));
    if (!result) {
        yyjson_doc_free(doc);
        return -1;
    }

    size_t count = 0;
    yyjson_val *item;
    yyjson_arr_iter iter;
    yyjson_arr_iter_init(siblings, &iter);
    while ((item = yyjson_arr_iter_next(&iter))) {
        yyjson_val *rf = yyjson_obj_get(item, "rfilename");
        const char *name = yyjson_get_str(rf);
        if (!name)
            continue;
        if (!reg_rfilename_safe(name))
            continue;
        if (!reg_file_wanted(name))
            continue;
        result[count] = dup_str(name);
        if (!result[count]) {
            reg_file_plan_free(result, count);
            yyjson_doc_free(doc);
            return -1;
        }
        count++;
    }

    yyjson_doc_free(doc);

    if (count == 0) {
        free(result);
        return 0;
    }

    *files = result;
    *n = count;
    return 0;
}

void reg_file_plan_free(char **files, size_t n) {
    if (!files)
        return;
    for (size_t i = 0; i < n; i++)
        free(files[i]);
    free(files);
}

static void curl_init_once(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    static int done = 0;
    if (!done) {
        pthread_once(&once, (void(*)(void))(void *)curl_global_init);
        done = 1;
    }
}

typedef struct {
    FILE                *fp;
    registry_progress_cb cb;
    void                *ud;
    size_t               idx;
    size_t               count;
    uint64_t             downloaded;
    uint64_t             total;
    int                  aborted;
} dl_ctx_t;

static size_t dl_write_cb(void *data, size_t size, size_t nmemb, void *userdata) {
    dl_ctx_t *ctx = (dl_ctx_t *)userdata;
    size_t total = size * nmemb;
    size_t written = fwrite(data, 1, total, ctx->fp);
    ctx->downloaded += written;
    return written;
}

static int dl_progress_cb(void *userdata, curl_off_t dltotal, curl_off_t dlnow,
                           curl_off_t ultotal, curl_off_t ulnow) {
    (void)ultotal;
    (void)ulnow;
    dl_ctx_t *ctx = (dl_ctx_t *)userdata;
    if (!ctx->cb)
        return 0;
    registry_progress_t p = {0};
    p.file_index = ctx->idx;
    p.file_count = ctx->count;
    p.file_downloaded = (uint64_t)dlnow;
    p.file_total = dltotal > 0 ? (uint64_t)dltotal : 0;
    if (ctx->cb(&p, ctx->ud) != 0) {
        ctx->aborted = 1;
        return 1;
    }
    return 0;
}

int reg_download_file(const char *url, const char *dest, const char *token,
                      registry_progress_cb cb, void *ud, size_t idx, size_t count) {
    curl_init_once();

    char part_path[4096];
    snprintf(part_path, sizeof(part_path), "%s.part", dest);

    struct stat part_st;
    long resume_from = 0;
    int has_part = (stat(part_path, &part_st) == 0 && S_ISREG(part_st.st_mode));
    if (has_part)
        resume_from = (long)part_st.st_size;

    FILE *fp = fopen(part_path, has_part ? "ab" : "wb");
    if (!fp) {
        log_error("cannot open %s for writing", part_path);
        return -1;
    }

    dl_ctx_t ctx = {0};
    ctx.fp = fp;
    ctx.cb = cb;
    ctx.ud = ud;
    ctx.idx = idx;
    ctx.count = count;

    CURL *c = curl_easy_init();
    if (!c) {
        fclose(fp);
        return -1;
    }

    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, dl_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, dl_progress_cb);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, &ctx);

    if (resume_from > 0)
        curl_easy_setopt(c, CURLOPT_RESUME_FROM_LARGE, (curl_off_t)resume_from);

    struct curl_slist *hdrs = NULL;
    if (token && token[0]) {
        char auth_hdr[512];
        snprintf(auth_hdr, sizeof(auth_hdr), "Authorization: Bearer %s", token);
        hdrs = curl_slist_append(hdrs, auth_hdr);
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
    }

    CURLcode res = curl_easy_perform(c);
    fclose(fp);

    long http_code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(c);
    if (hdrs) curl_slist_free_all(hdrs);

    if (ctx.aborted) {
        return -1;
    }

    if (res != CURLE_OK && res != CURLE_HTTP_RETURNED_ERROR) {
        log_error("download failed: %s", curl_easy_strerror(res));
        remove(part_path);
        return -1;
    }

    if (http_code == 200 && resume_from > 0) {
        fclose(fopen(part_path, "wb"));
        fp = fopen(part_path, "wb");
        if (!fp) {
            remove(part_path);
            return -1;
        }
        ctx.fp = fp;
        ctx.downloaded = 0;

        CURL *c2 = curl_easy_init();
        curl_easy_setopt(c2, CURLOPT_URL, url);
        curl_easy_setopt(c2, CURLOPT_WRITEFUNCTION, dl_write_cb);
        curl_easy_setopt(c2, CURLOPT_WRITEDATA, &ctx);
        curl_easy_setopt(c2, CURLOPT_FOLLOWLOCATION, 1L);
        if (token && token[0]) {
            char auth_hdr2[512];
            snprintf(auth_hdr2, sizeof(auth_hdr2), "Authorization: Bearer %s", token);
            hdrs = curl_slist_append(NULL, auth_hdr2);
            curl_easy_setopt(c2, CURLOPT_HTTPHEADER, hdrs);
        }
        res = curl_easy_perform(c2);
        fclose(fp);
        curl_easy_getinfo(c2, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(c2);
        if (hdrs) curl_slist_free_all(hdrs);
    }

    if (http_code == 416) {
        rename(part_path, dest);
        return 0;
    }

    if (http_code < 200 || http_code >= 300) {
        if (http_code == 401 || http_code == 403)
            log_error("model may be gated; set HF_TOKEN");
        remove(part_path);
        return -1;
    }

    if (rename(part_path, dest) != 0) {
        log_error("failed to rename %s to %s", part_path, dest);
        return -1;
    }

    return 0;
}
