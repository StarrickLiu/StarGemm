#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include <cute/tensor.hpp>

using namespace cute;

int main(int argc, char *argv[]) {
    auto smem_atom = composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<32>{}, Int<32>{}),
                                        make_stride(Int<32>{}, Int<1>{})));
    print_latex(smem_atom);
}