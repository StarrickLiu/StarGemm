#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include <cute/tensor.hpp>

using namespace cute;

int main(int argc, char *argv[]) {
    Layout layout = make_layout(make_shape(Int<32>(), Int<4>{}),
                                make_stride(Int<4>{}, Int<1>{}));
    print_layout(layout);
}