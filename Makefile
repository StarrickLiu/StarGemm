CC_FILES=$(shell find ./src/ -name "*.cu")
EXE_FILES=$(patsubst ./src/%.cu,./build/%,$(CC_FILES))

all: build $(EXE_FILES)

build:
	@mkdir -p build

./build/%: ./src/%.cu
	nvcc -o $@ $< -O2 -arch=sm_86 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas

clean:
	rm -rf build

