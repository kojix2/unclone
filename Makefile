CRYSTAL_BIN := bin/unclone
CRYSTAL_ENTRYPOINT := src/main.cr
RUST_DIR := rust-kernel
RUST_LIB_DIR := $(RUST_DIR)/target/release
RUST_LIB := libpcv_kernel.a
RUST_LINK_INPUT := $(CURDIR)/$(RUST_LIB_DIR)/$(RUST_LIB)
BASE_CRYSTAL_LINK_FLAGS := -L$(CURDIR)/$(RUST_LIB_DIR) $(RUST_LINK_INPUT)
PLATFORM_CRYSTAL_LINK_FLAGS :=
release ?= 0
cpu ?=
CARGO_BUILD_FLAGS := --release
CRYSTAL_RELEASE_FLAGS :=

ifeq ($(release),1)
CRYSTAL_RELEASE_FLAGS += --release
endif

ifneq ($(cpu),)
override RUSTFLAGS += -C target-cpu=$(cpu)
export RUSTFLAGS
endif

ifeq ($(OS),Windows_NT)
WIN_CURDIR := $(shell cygpath -m "$(CURDIR)")
RUST_LINK_INPUT := $(WIN_CURDIR)/$(RUST_LIB_DIR)/$(RUST_LIB)
BASE_CRYSTAL_LINK_FLAGS := -L$(WIN_CURDIR)/$(RUST_LIB_DIR) $(RUST_LINK_INPUT)
PLATFORM_CRYSTAL_LINK_FLAGS += -lbcrypt -luserenv
endif

EXTRA_CRYSTAL_LINK_FLAGS ?=
EXTRA_CRYSTAL_BUILD_FLAGS ?=
CRYSTAL_LINK_FLAGS := $(strip $(BASE_CRYSTAL_LINK_FLAGS) $(PLATFORM_CRYSTAL_LINK_FLAGS) $(EXTRA_CRYSTAL_LINK_FLAGS))
CRYSTAL_BUILD_FLAGS := $(strip $(CRYSTAL_RELEASE_FLAGS) $(EXTRA_CRYSTAL_BUILD_FLAGS))

.PHONY: all help build build-rust build-crystal test fmt lint clean

all: build

help:
	@echo "unclone Make targets"
	@echo "  make help         Show this help"
	@echo "  make build        Build Rust kernel and Crystal CLI"
	@echo "  make build release=1"
	@echo "                    Build Rust kernel and Crystal CLI in release mode"
	@echo "  make build release=1 cpu=native"
	@echo "                    Build a local CPU-tuned binary"
	@echo "  make test         Run Rust tests and Crystal specs"
	@echo "  make fmt          Format Rust and Crystal source"
	@echo "  make lint         Run Rust clippy and Crystal Ameba linters"
	@echo "  make clean        Remove build artifacts"

build: build-rust build-crystal

build-rust:
	cargo build --manifest-path $(RUST_DIR)/Cargo.toml $(CARGO_BUILD_FLAGS)

build-crystal: build-rust
	mkdir -p bin
	crystal build $(CRYSTAL_BUILD_FLAGS) $(CRYSTAL_ENTRYPOINT) -o $(CRYSTAL_BIN) --link-flags "$(CRYSTAL_LINK_FLAGS)"

test: build-rust
	cargo test --manifest-path $(RUST_DIR)/Cargo.toml
	crystal spec --link-flags "$(CRYSTAL_LINK_FLAGS)"

fmt:
	cargo fmt --manifest-path $(RUST_DIR)/Cargo.toml
	crystal tool format $(CRYSTAL_ENTRYPOINT) src/ spec/

lint:
	cargo clippy --manifest-path $(RUST_DIR)/Cargo.toml -- -D warnings
	lib/ameba/bin/ameba

clean:
	rm -rf bin
	cargo clean --manifest-path $(RUST_DIR)/Cargo.toml
