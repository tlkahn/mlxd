CC       ?= cc
CXX      ?= c++
CFLAGS   := -std=c11 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L -Isrc -Ivendor
CXXFLAGS := -std=c++17 -O2 -DNDEBUG -Ivendor/jinja_cpp
LDFLAGS  :=

PREFIX   ?= /usr/local

# --- Dependencies -------------------------------------------------------------

# Homebrew prefix (arm64 default; override for x86_64)
BREW_PREFIX   ?= /opt/homebrew

# mlx-c has no .pc file; use Homebrew paths directly
MLX_C_CFLAGS  := -I$(BREW_PREFIX)/include
MLX_C_LIBS    := -L$(BREW_PREFIX)/lib -lmlxc

# libuv and llhttp via pkg-config
LIBUV_CFLAGS  := $(shell pkg-config --cflags libuv 2>/dev/null)
LIBUV_LIBS    := $(shell pkg-config --libs   libuv 2>/dev/null)
LLHTTP_PC     := $(shell pkg-config --exists llhttp && echo llhttp || echo libllhttp)
LLHTTP_CFLAGS := $(shell pkg-config --cflags $(LLHTTP_PC) 2>/dev/null)
LLHTTP_LIBS   := $(shell pkg-config --libs   $(LLHTTP_PC) 2>/dev/null)
CURL_CFLAGS   := $(shell pkg-config --cflags libcurl 2>/dev/null)
CURL_LIBS     := $(shell pkg-config --libs   libcurl 2>/dev/null)

FRAMEWORKS := -framework Metal -framework Foundation -framework IOKit -framework CoreFoundation

ALL_CFLAGS  := $(CFLAGS) $(MLX_C_CFLAGS) $(LIBUV_CFLAGS) $(LLHTTP_CFLAGS) $(CURL_CFLAGS)
ALL_LDFLAGS := $(LDFLAGS) $(MLX_C_LIBS) $(LIBUV_LIBS) $(LLHTTP_LIBS) $(CURL_LIBS) $(FRAMEWORKS) -lc++

# --- Sources ------------------------------------------------------------------

SRCS := $(wildcard src/*.c src/*/*.c)
OBJS := $(SRCS:.c=.o) vendor/yyjson/yyjson.o

JINJA_SRCS := $(wildcard vendor/jinja_cpp/*.cpp)
JINJA_OBJS := $(JINJA_SRCS:.cpp=.o)

ALL_OBJS := $(OBJS) $(JINJA_OBJS)

# Auto-dependency tracking
DEPFLAGS = -MMD -MP
DEPS     := $(OBJS:.o=.d)

# --- Main target --------------------------------------------------------------

mlxd: $(ALL_OBJS)
	$(CC) -o $@ $^ $(ALL_LDFLAGS)

# --- Compile rules ------------------------------------------------------------

%.o: %.c
	$(CC) $(ALL_CFLAGS) $(DEPFLAGS) -c -o $@ $<

vendor/yyjson/yyjson.o: vendor/yyjson/yyjson.c
	$(CC) -std=c11 -O2 -DNDEBUG -c -o $@ $<

vendor/jinja_cpp/%.o: vendor/jinja_cpp/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

-include $(DEPS)

# --- Tests --------------------------------------------------------------------

TEST_SRCS      := $(filter-out tests/test_%_gpu.c, $(wildcard tests/test_*.c))
TEST_BINS      := $(TEST_SRCS:.c=)
TEST_GPU_SRCS  := $(wildcard tests/test_*_gpu.c)
TEST_GPU_BINS  := $(TEST_GPU_SRCS:.c=)
LIB_OBJS       := $(filter-out src/main.o, $(ALL_OBJS))

# test_http_gen_request includes gen_request.c directly with mock conn_io;
# must NOT link server.o, gen_request.o, or handler.o.
GENREQ_EXCL    := src/http/server.o src/http/gen_request.o src/http/handler.o
GENREQ_OBJS    := $(filter-out $(GENREQ_EXCL), $(LIB_OBJS))
tests/test_http_gen_request: tests/test_http_gen_request.c $(GENREQ_OBJS)
	$(CC) $(ALL_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -o $@ $< $(GENREQ_OBJS) $(ALL_LDFLAGS)

tests/test_%: tests/test_%.c $(LIB_OBJS)
	$(CC) $(ALL_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -o $@ $< $(LIB_OBJS) $(ALL_LDFLAGS)

test: $(TEST_BINS) test-parity-skip
	@pass=0; fail=0; \
	for t in $(TEST_BINS); do \
		printf "  %-40s" "$$(basename $$t)"; \
		if ./$$t > /dev/null 2>&1; then \
			printf "OK\n"; pass=$$((pass + 1)); \
		else \
			printf "FAIL\n"; fail=$$((fail + 1)); \
		fi; \
	done; \
	printf "\n%d passed, %d failed\n" $$pass $$fail; \
	[ $$fail -eq 0 ]

test-gpu: $(TEST_GPU_BINS) fixtures-tiny-ckpt
	@pass=0; fail=0; \
	for t in $(TEST_GPU_BINS); do \
		printf "  %-40s" "$$(basename $$t)"; \
		if ./$$t > /dev/null 2>&1; then \
			printf "OK\n"; pass=$$((pass + 1)); \
		else \
			printf "FAIL\n"; fail=$$((fail + 1)); \
		fi; \
	done; \
	printf "\n%d passed, %d failed\n" $$pass $$fail; \
	[ $$fail -eq 0 ]

# --- Housekeeping -------------------------------------------------------------

# --- Fixture generators -------------------------------------------------------

tools/gen_tiny_qwen3_ckpt: tools/gen_tiny_qwen3_ckpt.c vendor/yyjson/yyjson.o
	$(CC) $(ALL_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -o $@ $< vendor/yyjson/yyjson.o $(ALL_LDFLAGS)

fixtures-tiny-ckpt: tools/gen_tiny_qwen3_ckpt
	./tools/gen_tiny_qwen3_ckpt

# Regenerate src/model/tok_unicode_tables.{h,c} from Python's bundled UCD.
# Both files are checked in; normal builds never run this.
unicode-tables:
	python3 tools/gen_unicode_tables.py

clean:
	rm -f mlxd $(ALL_OBJS) $(DEPS) $(TEST_BINS) $(TEST_GPU_BINS) tests/test_*.d tools/gen_tiny_qwen3_ckpt
	find src vendor -name '*.tsan.o' -delete
	rm -rf build/tsan

install: mlxd
	install -d $(PREFIX)/bin
	install -m 755 mlxd $(PREFIX)/bin/mlxd

compile_commands.json: Makefile
	bear -- $(MAKE) clean mlxd

test-leaks: tests/test_http_server
	leaks --atExit -- ./tests/test_http_server

test-parity-skip:
	@out=$$(env -u MLXD_MLX_SERVE_BIN sh scripts/parity_temp0.sh 2>&1) && \
		printf '%s\n' "$$out" | grep -q 'skipped' || \
		{ printf "  %-40sFAIL (env-unset skip)\n" "parity-skip"; exit 1; }
	@out=$$(MLXD_MLX_SERVE_BIN=/nonexistent sh scripts/parity_temp0.sh /nonexistent-ckpt 2>&1) && \
		printf '%s\n' "$$out" | grep -q 'skipped' || \
		{ printf "  %-40sFAIL (bad-bin skip)\n" "parity-skip"; exit 1; }
	@tmpdir=$$(mktemp -d); \
	mkdir -p "$$tmpdir/scripts"; \
	cp scripts/parity_temp0.sh "$$tmpdir/scripts/"; \
	printf '#!/bin/sh\nexit 0\n' > "$$tmpdir/stub"; chmod +x "$$tmpdir/stub"; \
	mkdir -p "$$tmpdir/ckpt"; \
	out=$$(MLXD_MLX_SERVE_BIN="$$tmpdir/stub" sh "$$tmpdir/scripts/parity_temp0.sh" "$$tmpdir/ckpt" 2>&1) && rc=0 || rc=$$?; \
	rm -rf "$$tmpdir"; \
	if [ "$$rc" -eq 0 ]; then printf "  %-40sFAIL (build-fail: expected nonzero exit)\n" "parity-skip"; exit 1; fi; \
	printf '%s\n' "$$out" | grep -q 'build failed' || \
	{ printf "  %-40sFAIL (build-fail: no error message)\n" "parity-skip"; exit 1; }
	@tmpstub=$$(mktemp); printf '#!/bin/sh\nexit 0\n' > "$$tmpstub"; chmod +x "$$tmpstub"; \
	out=$$(MLXD_MLX_SERVE_BIN="$$tmpstub" sh scripts/parity_temp0.sh 2>&1); \
	rm -f "$$tmpstub"; \
	printf '%s\n' "$$out" | grep -q 'skipped: checkpoint dir' || \
	{ printf "  %-40sFAIL (ckpt-missing skip)\n" "parity-skip"; exit 1; }
	@printf "  %-40sOK\n" "parity-skip"

test-parity-script:
	@sh tests/test_parity_script.sh

.PHONY: test test-gpu test-tsan test-leaks test-parity-skip test-parity-script clean install analyze coverage coverage-gpu clean-coverage unicode-tables fixtures-tiny-ckpt

# --- Thread Sanitizer tests ---------------------------------------------------

TSAN_CFLAGS   := -g -O1 -fsanitize=thread
TSAN_LDFLAGS  := -fsanitize=thread
TSAN_DIR      := build/tsan
TSAN_ALL_OBJS := $(SRCS:.c=.tsan.o)
JINJA_TSAN_OBJS := $(JINJA_SRCS:.cpp=.tsan.o)
TSAN_LIB_OBJS := $(filter-out src/main.tsan.o,$(TSAN_ALL_OBJS)) vendor/yyjson/yyjson.tsan.o $(JINJA_TSAN_OBJS)

.PRECIOUS: %.tsan.o
%.tsan.o: %.c
	$(CC) $(ALL_CFLAGS) $(TSAN_CFLAGS) -c -o $@ $<

vendor/yyjson/yyjson.tsan.o: vendor/yyjson/yyjson.c
	$(CC) -std=c11 -g -O1 -DNDEBUG -fsanitize=thread -c -o $@ $<

vendor/jinja_cpp/%.tsan.o: vendor/jinja_cpp/%.cpp
	$(CXX) $(CXXFLAGS) -g -O1 -fsanitize=thread -c -o $@ $<

GENREQ_TSAN_EXCL := src/http/server.tsan.o src/http/gen_request.tsan.o src/http/handler.tsan.o
GENREQ_TSAN_OBJS := $(filter-out $(GENREQ_TSAN_EXCL), $(TSAN_LIB_OBJS))
$(TSAN_DIR)/test_http_gen_request: tests/test_http_gen_request.c $(GENREQ_TSAN_OBJS) | $(TSAN_DIR)
	$(CC) $(ALL_CFLAGS) $(TSAN_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -o $@ $< $(GENREQ_TSAN_OBJS) $(ALL_LDFLAGS) $(TSAN_LDFLAGS)

$(TSAN_DIR)/test_%: tests/test_%.c $(TSAN_LIB_OBJS) | $(TSAN_DIR)
	$(CC) $(ALL_CFLAGS) $(TSAN_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -o $@ $< $(TSAN_LIB_OBJS) $(ALL_LDFLAGS) $(TSAN_LDFLAGS)

$(TSAN_DIR):
	mkdir -p $@

TSAN_TEST_BINS := $(TEST_SRCS:tests/test_%.c=$(TSAN_DIR)/test_%)

test-tsan: $(TSAN_TEST_BINS)
	@pass=0; fail=0; \
	for t in $(TSAN_TEST_BINS); do \
		printf "  %-40s" "$$(basename $$t)"; \
		if ./$$t > /dev/null 2>&1; then \
			printf "OK\n"; pass=$$((pass + 1)); \
		else \
			printf "FAIL\n"; fail=$$((fail + 1)); \
		fi; \
	done; \
	printf "\n%d passed, %d failed (tsan)\n" $$pass $$fail; \
	[ $$fail -eq 0 ]

# --- Debug/Release shortcuts --------------------------------------------------

debug: CFLAGS += -g -O0 -DDEBUG -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: mlxd

tsan: CFLAGS += -g -O1 -DDEBUG -fsanitize=thread
tsan: LDFLAGS += -fsanitize=thread
tsan: mlxd

release: CFLAGS += -O2 -DNDEBUG
release: mlxd

SCAN_BUILD ?= $(shell command -v scan-build 2>/dev/null || echo $(BREW_PREFIX)/opt/llvm/bin/scan-build)

analyze:
	$(SCAN_BUILD) --use-cc=$(CC) --status-bugs --exclude vendor $(MAKE) clean mlxd

# --- Coverage (clang source-based) --------------------------------------------

LLVM_PREFIX   ?= $(BREW_PREFIX)/opt/llvm
LLVM_PROFDATA := $(LLVM_PREFIX)/bin/llvm-profdata
LLVM_COV      := $(LLVM_PREFIX)/bin/llvm-cov
COV_DIR       := build/cov

COV_CFLAGS  := -fprofile-instr-generate -fcoverage-mapping
COV_LDFLAGS := -fprofile-instr-generate

COV_ALL_OBJS := $(SRCS:.c=.cov.o)
COV_LIB_OBJS := $(filter-out src/main.cov.o, $(COV_ALL_OBJS)) vendor/yyjson/yyjson.o $(JINJA_OBJS)

.PRECIOUS: %.cov.o
%.cov.o: %.c
	$(CC) $(ALL_CFLAGS) $(COV_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -c -o $@ $<

COV_TEST_SRCS     := $(filter-out tests/test_%_gpu.c, $(wildcard tests/test_*.c))
COV_TEST_BINS     := $(COV_TEST_SRCS:tests/test_%.c=$(COV_DIR)/test_%)

COV_GPU_TEST_SRCS := $(wildcard tests/test_*_gpu.c)
COV_GPU_TEST_BINS := $(COV_GPU_TEST_SRCS:tests/test_%.c=$(COV_DIR)/test_%)

GENREQ_COV_EXCL := src/http/server.cov.o src/http/gen_request.cov.o src/http/handler.cov.o
GENREQ_COV_OBJS := $(filter-out $(GENREQ_COV_EXCL), $(COV_LIB_OBJS))
$(COV_DIR)/test_http_gen_request: tests/test_http_gen_request.c $(GENREQ_COV_OBJS) | $(COV_DIR)
	$(CC) $(ALL_CFLAGS) $(COV_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -o $@ $< $(GENREQ_COV_OBJS) $(ALL_LDFLAGS) $(COV_LDFLAGS)

$(COV_DIR)/test_%: tests/test_%.c $(COV_LIB_OBJS) | $(COV_DIR)
	$(CC) $(ALL_CFLAGS) $(COV_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -o $@ $< $(COV_LIB_OBJS) $(ALL_LDFLAGS) $(COV_LDFLAGS)

$(COV_DIR):
	mkdir -p $@

coverage: $(COV_TEST_BINS) | $(COV_DIR)
	@rm -f $(COV_DIR)/*.profraw $(COV_DIR)/merged.profdata
	@for t in $(COV_TEST_BINS); do \
		LLVM_PROFILE_FILE="$(COV_DIR)/$$(basename $$t).profraw" ./$$t > /dev/null 2>&1 || true; \
	done
	$(LLVM_PROFDATA) merge -sparse $(COV_DIR)/*.profraw -o $(COV_DIR)/merged.profdata
	@echo ""
	@echo "--- Coverage summary ---"
	@$(LLVM_COV) report $(firstword $(COV_TEST_BINS)) \
		$(addprefix -object ,$(wordlist 2,$(words $(COV_TEST_BINS)),$(COV_TEST_BINS))) \
		-instr-profile=$(COV_DIR)/merged.profdata \
		-ignore-filename-regex='vendor/|tests/'
	$(LLVM_COV) show $(firstword $(COV_TEST_BINS)) \
		$(addprefix -object ,$(wordlist 2,$(words $(COV_TEST_BINS)),$(COV_TEST_BINS))) \
		-instr-profile=$(COV_DIR)/merged.profdata \
		-format=html -output-dir=$(COV_DIR)/html \
		-ignore-filename-regex='vendor/|tests/'
	@echo ""
	@echo "HTML report: $(COV_DIR)/html/index.html"

coverage-gpu: $(COV_GPU_TEST_BINS) fixtures-tiny-ckpt | $(COV_DIR)
	@rm -f $(COV_DIR)/gpu-*.profraw $(COV_DIR)/gpu-merged.profdata
	@for t in $(COV_GPU_TEST_BINS); do \
		LLVM_PROFILE_FILE="$(COV_DIR)/gpu-$$(basename $$t).profraw" ./$$t > /dev/null 2>&1 || true; \
	done
	$(LLVM_PROFDATA) merge -sparse $(COV_DIR)/gpu-*.profraw -o $(COV_DIR)/gpu-merged.profdata
	@echo ""
	@echo "--- GPU coverage summary ---"
	@$(LLVM_COV) report $(firstword $(COV_GPU_TEST_BINS)) \
		$(addprefix -object ,$(wordlist 2,$(words $(COV_GPU_TEST_BINS)),$(COV_GPU_TEST_BINS))) \
		-instr-profile=$(COV_DIR)/gpu-merged.profdata \
		-ignore-filename-regex='vendor/|tests/'
	$(LLVM_COV) show $(firstword $(COV_GPU_TEST_BINS)) \
		$(addprefix -object ,$(wordlist 2,$(words $(COV_GPU_TEST_BINS)),$(COV_GPU_TEST_BINS))) \
		-instr-profile=$(COV_DIR)/gpu-merged.profdata \
		-format=html -output-dir=$(COV_DIR)/html-gpu \
		-ignore-filename-regex='vendor/|tests/'
	@echo ""
	@echo "HTML report: $(COV_DIR)/html-gpu/index.html"

clean-coverage:
	rm -rf build/cov
	find src -name '*.cov.o' -delete
