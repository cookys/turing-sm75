NVCC ?= nvcc
ARCH ?= sm_75
SRC  := src/attn_sm75.cu
BIN  := build/attn_sm75
GDN_SRC := src/gdn_sm75.cu
GDN_BIN := build/gdn_sm75

.PHONY: all run gdn sass clean

all: $(BIN) $(GDN_BIN)

build:
	mkdir -p build

ifeq ($(filter $(ARCH),sm_75),)
$(error ARCH must be sm_75; got $(ARCH). A different cubin is a different claim.)
endif

$(BIN): $(SRC) | build
	$(NVCC) -O3 -std=c++17 -gencode arch=compute_75,code=$(ARCH) --cudart=static \
		-Xptxas=-v $< -o $@

$(GDN_BIN): $(GDN_SRC) | build
	$(NVCC) -O3 -std=c++17 -gencode arch=compute_75,code=$(ARCH) --cudart=static \
		-Xptxas=-v $< -o $@

run: $(BIN)
	./$(BIN) 1 4 256 128

gdn: $(GDN_BIN)
	./$(GDN_BIN) 512

sass: $(BIN)
	@echo "=== flash_attn_sm75_wmma (need HMMA) ==="
	cuobjdump -sass -fun flash_attn_sm75_wmma $(BIN) | grep -c HMMA
	@echo "=== no LDGSTS anywhere ==="
	@if cuobjdump -sass $(BIN) | grep -q LDGSTS; then echo "FAIL: LDGSTS"; exit 1; fi
	@echo OK

clean:
	rm -rf build
