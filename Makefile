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
LLHTTP_CFLAGS := $(shell pkg-config --cflags llhttp 2>/dev/null)
LLHTTP_LIBS   := $(shell pkg-config --libs   llhttp 2>/dev/null)

FRAMEWORKS := -framework Metal -framework Foundation -framework IOKit -framework CoreFoundation

ALL_CFLAGS  := $(CFLAGS) $(MLX_C_CFLAGS) $(LIBUV_CFLAGS) $(LLHTTP_CFLAGS)
ALL_LDFLAGS := $(LDFLAGS) $(MLX_C_LIBS) $(LIBUV_LIBS) $(LLHTTP_LIBS) $(FRAMEWORKS) -lc++

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

TEST_SRCS  := $(wildcard tests/test_*.c)
TEST_BINS  := $(TEST_SRCS:.c=)
LIB_OBJS   := $(filter-out src/main.o, $(ALL_OBJS))

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

# --- Housekeeping -------------------------------------------------------------

clean:
	rm -f mlxd $(ALL_OBJS) $(DEPS) $(TEST_BINS) tests/test_*.d

install: mlxd
	install -d $(PREFIX)/bin
	install -m 755 mlxd $(PREFIX)/bin/mlxd

.PHONY: test clean install analyze

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
	$(SCAN_BUILD) --use-cc=$(CC) --status-bugs $(MAKE) clean mlxd
