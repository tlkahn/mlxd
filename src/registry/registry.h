#ifndef MLXD_REGISTRY_H
#define MLXD_REGISTRY_H

/* Download/resolve models from HuggingFace.
 * Returns the local path on success, NULL on error.
 * Caller must free the returned string. */
char *registry_pull(const char *model_id);

/* Resolve a model specifier (name:tag or local path) to a local directory.
 * Returns a heap-allocated path, or NULL if not found. Caller must free. */
char *registry_resolve(const char *specifier);

#endif
