#include "registry/registry_internal.h"
#include "core/log.h"

#include <curl/curl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <yyjson/yyjson.h>

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
        result[count] = reg_dup_str(name);
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

static void curl_once_fn(void) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    atexit(curl_global_cleanup);
}

void reg_curl_init_once(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, curl_once_fn);
}

typedef struct {
    FILE                *fp;
    registry_progress_cb cb;
    void                *ud;
    size_t               idx;
    size_t               count;
    const char          *filename;
    int                  aborted;
    size_t               resume_base;
    CURL                *curl;
    const char          *part_path;
    int                  response_checked;
    long                 content_range_total;
} dl_ctx_t;

static size_t dl_write_cb(void *data, size_t size, size_t nmemb, void *userdata) {
    dl_ctx_t *ctx = (dl_ctx_t *)userdata;
    size_t total = size * nmemb;
    if (!ctx->response_checked && ctx->resume_base > 0) {
        ctx->response_checked = 1;
        long code = 0;
        curl_easy_getinfo(ctx->curl, CURLINFO_RESPONSE_CODE, &code);
        if (code == 200) {
            FILE *nfp = freopen(ctx->part_path, "wb", ctx->fp);
            if (!nfp)
                return 0;
            ctx->fp = nfp;
            ctx->resume_base = 0;
        }
    }
    return fwrite(data, 1, total, ctx->fp);
}

static int dl_progress_cb(void *userdata, curl_off_t dltotal, curl_off_t dlnow,
                           curl_off_t ultotal, curl_off_t ulnow) {
    (void)ultotal;
    (void)ulnow;
    dl_ctx_t *ctx = (dl_ctx_t *)userdata;
    if (!ctx->cb)
        return 0;
    registry_progress_t p = {0};
    p.filename = ctx->filename;
    p.file_index = ctx->idx;
    p.file_count = ctx->count;
    p.file_downloaded = (uint64_t)((curl_off_t)ctx->resume_base + dlnow);
    p.file_total = dltotal > 0 ? (uint64_t)((curl_off_t)ctx->resume_base + dltotal) : 0;
    if (ctx->cb(&p, ctx->ud) != 0) {
        ctx->aborted = 1;
        return 1;
    }
    return 0;
}

static size_t dl_header_cb(char *buf, size_t size, size_t nitems, void *userdata) {
    dl_ctx_t *ctx = (dl_ctx_t *)userdata;
    size_t total = size * nitems;
    if (total > 16 && strncasecmp(buf, "Content-Range:", 14) == 0) {
        long val = 0;
        if (sscanf(buf + 14, " bytes */%ld", &val) == 1 && val > 0)
            ctx->content_range_total = val;
    }
    return total;
}

int reg_download_file(const char *url, const char *dest, const char *token,
                      const char *filename,
                      registry_progress_cb cb, void *ud, size_t idx, size_t count) {
    reg_curl_init_once();

    char part_path[4096];
    snprintf(part_path, sizeof(part_path), "%s.part", dest);

    int attempt = 0;
retry:;
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
    ctx.filename = filename;
    ctx.resume_base = (size_t)(resume_from > 0 ? resume_from : 0);
    ctx.part_path = part_path;
    ctx.content_range_total = -1;

    CURL *c = curl_easy_init();
    if (!c) {
        fclose(fp);
        return -1;
    }
    ctx.curl = c;

    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, dl_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, dl_progress_cb);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, &ctx);
    curl_easy_setopt(c, CURLOPT_USERAGENT, MLXD_USER_AGENT);
    curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, dl_header_cb);
    curl_easy_setopt(c, CURLOPT_HEADERDATA, &ctx);

    struct curl_slist *hdrs = NULL;
    if (resume_from > 0) {
        char range_hdr[128];
        snprintf(range_hdr, sizeof(range_hdr), "Range: bytes=%ld-", resume_from);
        hdrs = curl_slist_append(hdrs, range_hdr);
    }
    if (token && token[0]) {
        char auth_hdr[512];
        snprintf(auth_hdr, sizeof(auth_hdr), "Authorization: Bearer %s", token);
        hdrs = curl_slist_append(hdrs, auth_hdr);
    }
    if (hdrs)
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);

    CURLcode res = curl_easy_perform(c);
    if (ctx.fp)
        fclose(ctx.fp);

    long http_code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(c);
    if (hdrs) {
        curl_slist_free_all(hdrs);
        hdrs = NULL;
    }

    if (ctx.aborted)
        return -1;

    if (res != CURLE_OK) {
        log_error("download failed: %s", curl_easy_strerror(res));
        return -1;
    }

    if (http_code == 416) {
        struct stat ps;
        if (ctx.content_range_total > 0 &&
            stat(part_path, &ps) == 0 &&
            ps.st_size == ctx.content_range_total) {
            rename(part_path, dest);
            return 0;
        }
        remove(part_path);
        if (attempt == 0) {
            attempt = 1;
            goto retry;
        }
        log_error("416 after retry");
        return -1;
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
