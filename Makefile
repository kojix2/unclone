CRYSTAL_BIN := bin/toyclone
RUST_DIR := rust-kernel
RUST_LIB_DIR := $(RUST_DIR)/target/release
RUST_LIB := libpcv_kernel.a
RUST_LINK_INPUT := $(CURDIR)/$(RUST_LIB_DIR)/$(RUST_LIB)
BASE_CRYSTAL_LINK_FLAGS := -L$(CURDIR)/$(RUST_LIB_DIR) $(RUST_LINK_INPUT)
PLATFORM_CRYSTAL_LINK_FLAGS :=

ifeq ($(OS),Windows_NT)
RUST_LINK_INPUT := -lpcv_kernel
BASE_CRYSTAL_LINK_FLAGS := -L$(CURDIR)/$(RUST_LIB_DIR) $(RUST_LINK_INPUT)
PLATFORM_CRYSTAL_LINK_FLAGS += -lbcrypt -luserenv
endif

EXTRA_CRYSTAL_LINK_FLAGS ?=
CRYSTAL_LINK_FLAGS := $(BASE_CRYSTAL_LINK_FLAGS) $(PLATFORM_CRYSTAL_LINK_FLAGS) $(EXTRA_CRYSTAL_LINK_FLAGS)

.PHONY: all help build build-rust build-crystal test clean run

all: build

help:
	@echo "ToyClone Make targets"
	@echo "  make help         Show this help"
	@echo "  make build        Build Rust kernel and Crystal CLI"
	@echo "  make test         Run Rust tests and Crystal specs"
	@echo "  make run ARGS='fit -i INPUT -o OUTPUT [options]'"
	@echo "                    Run the CLI with provided ARGS"
	@echo "  make clean        Remove build artifacts"

build: build-rust build-crystal

build-rust:
	cargo build --manifest-path $(RUST_DIR)/Cargo.toml --release

build-crystal: build-rust
	mkdir -p bin
	crystal build src/bin/pyclone-vi-cr.cr -o $(CRYSTAL_BIN) --link-flags "$(CRYSTAL_LINK_FLAGS)"

test: build-rust
	cargo test --manifest-path $(RUST_DIR)/Cargo.toml
	crystal spec --link-flags "$(CRYSTAL_LINK_FLAGS)"

run: build
	./$(CRYSTAL_BIN) $(ARGS)

clean:
	rm -rf bin
	cargo clean --manifest-path $(RUST_DIR)/Cargo.toml
