#include "registry/registry.h"
#include "registry/registry_internal.h"
#include "core/log.h"

#include <curl/curl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <yyjson/yyjson.h>

char *reg_dup_str(const char *s) {
    if (!s)
        return NULL;
    size_t n = strlen(s);
    char  *p = malloc(n + 1);
    if (p)
        memcpy(p, s, n + 1);
    return p;
}

char *reg_path_join(const char *dir, const char *name) {
    size_t dlen = strlen(dir);
    size_t nlen = strlen(name);
    char  *p    = malloc(dlen + 1 + nlen + 1);
    if (!p)
        return NULL;
    memcpy(p, dir, dlen);
    p[dlen] = '/';
    memcpy(p + dlen + 1, name, nlen);
    p[dlen + 1 + nlen] = '\0';
    return p;
}

int reg_dir_has_config_json(const char *dir) {
    char *p = reg_path_join(dir, "config.json");
    if (!p)
        return 0;
    struct stat st;
    int ok = (stat(p, &st) == 0 && S_ISREG(st.st_mode));
    free(p);
    return ok;
}

int reg_spec_parse(const char *spec, reg_spec_t *out) {
    if (!spec || !out)
        return -1;
    memset(out, 0, sizeof(*out));

    if (spec[0] == '\0')
        return -1;

    if (spec[0] == '.' || spec[0] == '/' || spec[0] == '~') {
        out->local_path = reg_dup_str(spec);
        return out->local_path ? 0 : -1;
    }

    const char *slash = strchr(spec, '/');
    if (!slash)
        return -1;

    size_t org_len = (size_t)(slash - spec);
    if (org_len == 0)
        return -1;

    const char *after_slash = slash + 1;
    if (strchr(after_slash, '/'))
        return -1;

    const char *colon = strchr(after_slash, ':');
    size_t model_len;
    if (colon) {
        model_len = (size_t)(colon - after_slash);
        const char *rev = colon + 1;
        if (rev[0] == '\0')
            return -1;
        out->revision = reg_dup_str(rev);
        if (!out->revision)
            return -1;
    } else {
        model_len = strlen(after_slash);
    }

    if (model_len == 0) {
        free(out->revision);
        out->revision = NULL;
        return -1;
    }

    out->org = malloc(org_len + 1);
    if (!out->org) {
        reg_spec_free(out);
        return -1;
    }
    memcpy(out->org, spec, org_len);
    out->org[org_len] = '\0';

    out->model = malloc(model_len + 1);
    if (!out->model) {
        reg_spec_free(out);
        return -1;
    }
    memcpy(out->model, after_slash, model_len);
    out->model[model_len] = '\0';

    return 0;
}

void reg_spec_free(reg_spec_t *s) {
    if (!s)
        return;
    free(s->org);
    free(s->model);
    free(s->revision);
    free(s->local_path);
    memset(s, 0, sizeof(*s));
}

char *reg_cache_root(void) {
    const char *env = getenv("MLXD_CACHE_DIR");
    if (env && env[0])
        return reg_dup_str(env);
    const char *home = getenv("HOME");
    if (!home)
        return NULL;
    size_t hlen = strlen(home);
    const char suffix[] = "/.cache/mlxd";
    char *p = malloc(hlen + sizeof(suffix));
    if (!p)
        return NULL;
    memcpy(p, home, hlen);
    memcpy(p + hlen, suffix, sizeof(suffix));
    return p;
}

char *reg_model_cache_dir(const char *org, const char *model) {
    char *root = reg_cache_root();
    if (!root)
        return NULL;
    size_t rlen = strlen(root);
    size_t olen = strlen(org);
    size_t mlen = strlen(model);
    char *p = malloc(rlen + 8 + olen + 2 + mlen + 1);
    if (!p) {
        free(root);
        return NULL;
    }
    memcpy(p, root, rlen);
    memcpy(p + rlen, "/models/", 8);
    memcpy(p + rlen + 8, org, olen);
    memcpy(p + rlen + 8 + olen, "--", 2);
    memcpy(p + rlen + 8 + olen + 2, model, mlen);
    p[rlen + 8 + olen + 2 + mlen] = '\0';
    free(root);
    return p;
}

char *reg_endpoint(void) {
    const char *env = getenv("HF_ENDPOINT");
    if (env && env[0])
        return reg_dup_str(env);
    return reg_dup_str("https://huggingface.co");
}

static const char *EXACT_FILES[] = {
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "generation_config.json", "special_tokens_map.json",
    "chat_template.jinja",
};
#define EXACT_FILES_N (sizeof(EXACT_FILES) / sizeof(EXACT_FILES[0]))

static int has_suffix(const char *s, const char *suffix) {
    size_t slen = strlen(s);
    size_t xlen = strlen(suffix);
    return slen >= xlen && strcmp(s + slen - xlen, suffix) == 0;
}

int reg_file_wanted(const char *rfilename) {
    if (!rfilename)
        return 0;
    for (size_t i = 0; i < EXACT_FILES_N; i++) {
        if (strcmp(rfilename, EXACT_FILES[i]) == 0)
            return 1;
    }
    if (has_suffix(rfilename, ".safetensors"))
        return 1;
    if (has_suffix(rfilename, ".safetensors.index.json"))
        return 1;
    return 0;
}

int reg_rfilename_safe(const char *rfilename) {
    if (!rfilename || rfilename[0] == '\0')
        return 0;
    if (rfilename[0] == '/')
        return 0;
    if (strstr(rfilename, ".."))
        return 0;
    if (strchr(rfilename, '\\'))
        return 0;
    return 1;
}

typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
} membuf_t;

static size_t membuf_write(void *data, size_t size, size_t nmemb, void *userdata) {
    membuf_t *m = (membuf_t *)userdata;
    size_t total = size * nmemb;
    if (m->len + total + 1 > m->cap) {
        size_t newcap = (m->cap + total + 1) * 2;
        char *tmp = realloc(m->buf, newcap);
        if (!tmp) return 0;
        m->buf = tmp;
        m->cap = newcap;
    }
    memcpy(m->buf + m->len, data, total);
    m->len += total;
    m->buf[m->len] = '\0';
    return total;
}

static int mkdirs(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
                return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
        return -1;
    return 0;
}

static int write_meta_json(const char *dir, const char *revision, const char *commit) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "revision", revision);
    if (commit)
        yyjson_mut_obj_add_str(doc, root, "commit", commit);

    char *json = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, NULL);
    yyjson_mut_doc_free(doc);
    if (!json)
        return -1;

    char *meta_path = reg_path_join(dir, ".mlxd-meta.json");
    if (!meta_path) {
        free(json);
        return -1;
    }

    FILE *f = fopen(meta_path, "w");
    free(meta_path);
    if (!f) {
        free(json);
        return -1;
    }
    fputs(json, f);
    fclose(f);
    free(json);
    return 0;
}

char *reg_meta_read_revision(const char *dir) {
    char *meta_path = reg_path_join(dir, ".mlxd-meta.json");
    if (!meta_path)
        return NULL;

    FILE *f = fopen(meta_path, "rb");
    free(meta_path);
    if (!f)
        return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(f);
        return NULL;
    }

    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);

    yyjson_doc *doc = yyjson_read(buf, (size_t)sz, 0);
    free(buf);
    if (!doc)
        return NULL;

    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *rev = yyjson_get_str(yyjson_obj_get(root, "revision"));
    char *result = rev ? reg_dup_str(rev) : NULL;
    yyjson_doc_free(doc);
    return result;
}

char *registry_pull(const char *spec, const registry_pull_opts_t *opts) {
    if (!spec)
        return NULL;

    reg_curl_init_once();

    reg_spec_t s = {0};
    if (reg_spec_parse(spec, &s) != 0)
        return NULL;

    if (s.local_path) {
        reg_spec_free(&s);
        return NULL;
    }

    const char *revision = "main";
    const char *token = NULL;
    registry_progress_cb cb = NULL;
    void *ud = NULL;

    if (opts) {
        if (opts->revision)
            revision = opts->revision;
        token = opts->token;
        cb = opts->progress;
        ud = opts->userdata;
    }
    if (s.revision)
        revision = s.revision;

    if (!token) {
        token = getenv("HF_TOKEN");
        if (token && !token[0])
            token = NULL;
    }

    char *endpoint = reg_endpoint();
    if (!endpoint) {
        reg_spec_free(&s);
        return NULL;
    }

    char api_url[2048];
    snprintf(api_url, sizeof(api_url), "%s/api/models/%s/%s/revision/%s",
             endpoint, s.org, s.model, revision);

    CURL *c = curl_easy_init();
    if (!c) {
        free(endpoint);
        reg_spec_free(&s);
        return NULL;
    }

    membuf_t api_buf = {0};
    api_buf.cap = 4096;
    api_buf.buf = calloc(1, api_buf.cap);

    curl_easy_setopt(c, CURLOPT_URL, api_url);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, membuf_write);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &api_buf);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_USERAGENT, MLXD_USER_AGENT);

    struct curl_slist *hdrs = NULL;
    if (token) {
        char auth_hdr[512];
        snprintf(auth_hdr, sizeof(auth_hdr), "Authorization: Bearer %s", token);
        hdrs = curl_slist_append(NULL, auth_hdr);
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
    }

    CURLcode res = curl_easy_perform(c);
    long http_code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(c);
    if (hdrs) {
        curl_slist_free_all(hdrs);
        hdrs = NULL;
    }

    if (res != CURLE_OK || http_code < 200 || http_code >= 300) {
        log_error("failed to fetch model info: HTTP %ld", http_code);
        free(api_buf.buf);
        free(endpoint);
        reg_spec_free(&s);
        return NULL;
    }

    char **files = NULL;
    size_t nfiles = 0;
    if (reg_parse_file_plan(api_buf.buf, api_buf.len, &files, &nfiles) != 0 || nfiles == 0) {
        log_error("no downloadable files in model");
        free(api_buf.buf);
        free(endpoint);
        reg_spec_free(&s);
        return NULL;
    }

    yyjson_doc *api_doc = yyjson_read(api_buf.buf, api_buf.len, 0);
    const char *commit_sha = NULL;
    if (api_doc) {
        yyjson_val *api_root = yyjson_doc_get_root(api_doc);
        commit_sha = yyjson_get_str(yyjson_obj_get(api_root, "sha"));
    }

    char *cache_dir = reg_model_cache_dir(s.org, s.model);
    if (!cache_dir || mkdirs(cache_dir) != 0) {
        log_error("failed to create cache directory");
        reg_file_plan_free(files, nfiles);
        if (api_doc) yyjson_doc_free(api_doc);
        free(api_buf.buf);
        free(endpoint);
        free(cache_dir);
        reg_spec_free(&s);
        return NULL;
    }

    int force_redownload = 0;
    char *cached_rev = reg_meta_read_revision(cache_dir);
    if (cached_rev) {
        if (strcmp(cached_rev, revision) != 0)
            force_redownload = 1;
        free(cached_rev);
    }

    int ok = 1;
    for (size_t i = 0; i < nfiles && ok; i++) {
        char *dest = reg_path_join(cache_dir, files[i]);
        if (!dest) {
            ok = 0;
            break;
        }

        if (!force_redownload) {
            struct stat fst;
            if (stat(dest, &fst) == 0 && S_ISREG(fst.st_mode) && fst.st_size > 0) {
                free(dest);
                continue;
            }
        }

        char file_url[2048];
        snprintf(file_url, sizeof(file_url), "%s/%s/%s/resolve/%s/%s",
                 endpoint, s.org, s.model, revision, files[i]);

        int rc = reg_download_file(file_url, dest, token, files[i], cb, ud, i, nfiles);
        free(dest);
        if (rc != 0) {
            ok = 0;
            break;
        }
    }

    reg_file_plan_free(files, nfiles);
    free(endpoint);

    if (!ok) {
        if (api_doc) yyjson_doc_free(api_doc);
        free(api_buf.buf);
        free(cache_dir);
        reg_spec_free(&s);
        return NULL;
    }

    write_meta_json(cache_dir, revision, commit_sha);

    if (api_doc) yyjson_doc_free(api_doc);
    free(api_buf.buf);
    reg_spec_free(&s);
    return cache_dir;
}

char *registry_resolve(const char *specifier) {
    if (!specifier)
        return NULL;

    reg_spec_t s = {0};
    if (reg_spec_parse(specifier, &s) != 0)
        return NULL;

    char *dir = NULL;
    if (s.local_path) {
        if (reg_dir_has_config_json(s.local_path))
            dir = reg_dup_str(s.local_path);
    } else {
        dir = reg_model_cache_dir(s.org, s.model);
        if (dir && !reg_dir_has_config_json(dir)) {
            free(dir);
            dir = NULL;
        }
        if (dir && s.revision) {
            char *cached_rev = reg_meta_read_revision(dir);
            if (!cached_rev || strcmp(cached_rev, s.revision) != 0) {
                free(dir);
                dir = NULL;
            }
            free(cached_rev);
        }
    }

    reg_spec_free(&s);
    return dir;
}
