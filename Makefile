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

$(BIN): $(SRC) | build
	$(NVCC) -O3 -std=c++17 -gencode arch=compute_75,code=$(ARCH) --cudart=static \
		-Xptxas=-v $< -o $@

$(GDN_BIN): $(GDN_SRC) | build
	$(NVCC) -O3 -std=c++17 -gencode arch=compute_75,code=$(ARCH) --cudart=static \
		-Xptxas=-v $< -o $@

run: $(BIN)
	./$(BIN) 1 4 256 128

gdn: $(GDN_BIN)
	./$(GDN_BIN) 64

sass: $(BIN)
	cuobjdump -sass $(BIN) | grep -E 'HMMA|IMMA|FFMA|LDGSTS' | head

clean:
	rm -rf build
