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

tests/test_%: tests/test_%.c $(LIB_OBJS)
	$(CC) $(ALL_CFLAGS) -DMLXD_FIXTURES_DIR=\"$(CURDIR)/tests/fixtures\" -o $@ $< $(LIB_OBJS) $(ALL_LDFLAGS)

test: $(TEST_BINS)
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

test-gpu: $(TEST_GPU_BINS)
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

# Regenerate src/model/tok_unicode_tables.{h,c} from Python's bundled UCD.
# Both files are checked in; normal builds never run this.
unicode-tables:
	python3 tools/gen_unicode_tables.py

clean:
	rm -f mlxd $(ALL_OBJS) $(DEPS) $(TEST_BINS) $(TEST_GPU_BINS) tests/test_*.d

install: mlxd
	install -d $(PREFIX)/bin
	install -m 755 mlxd $(PREFIX)/bin/mlxd

compile_commands.json: Makefile
	bear -- $(MAKE) clean mlxd

.PHONY: test test-gpu clean install analyze coverage clean-coverage unicode-tables

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

COV_TEST_SRCS := $(wildcard tests/test_*.c)
COV_TEST_BINS := $(COV_TEST_SRCS:tests/test_%.c=$(COV_DIR)/test_%)

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

clean-coverage:
	rm -rf build/cov
	find src -name '*.cov.o' -delete
