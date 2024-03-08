#include <cublas_v2.h>
#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include <cute/tensor.hpp>

#include "detail/cublaslt-gemm.h"
#include "detail/data.h"

template <typename Config>
__global__ void gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr, int m, int n, int k) {
    using namespace cute;

    // 从Config中获取信息
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;
    using TiledMMA = typename Config::MMA;
    
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyAtomC = typename Config::S2GCopyAtomC;
    using S2GCopyC = typename Config::S2GCopyC;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;
    
    extern __shared__ T shm_data[];

    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    // 用Tensor描述全局内存的矩阵内存位置

    Tensor A = make_tensor(make_gmem_ptr((T*)Aptr),
                           make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr((T*)Bptr),
                           make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr((T*)Dptr),
                           make_shape(m, n), make_stride(n, Int<1>{}));

    // 获取当前线程块处理的Tile

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                           make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                           make_coord(ix, _));
    Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                           make_coord(iy, ix));
    //  gA(128, 32, 8)
    //  gB(128, 32, 8)
    //  gD(128, 128) 

    // 使用Tensor描述ShareMemory上的A和B

    Tensor sA = make_tensor(make_smem_ptr(Ashm),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(Bshm),
                            SmemLayoutB{});
    
    // TiledMMA
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    // 按照partition类函数返回的Tensor形状生成的对应的寄存器声明
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    Tensor tCrD = thr_mma.partition_fragment_C(gD);
    // 分到的是每个线程自己的数据，Shape输出如下：
    //  tCrA(MMA, MMA_M, MMA_K) (8, 4, 2)
    //  tCrB(MMA, MMA_N, MMA_K) (4, 8, 2)
    //  tCrD(MMA, MMA_M, MMA_N) (4, 4, 8)
    // 现从理论角度分析为什么会得到这样的Shape
    //  已知TiledMMA的大小是 32*32*16 不考虑N维上的Atom重复运算（没有影响线程数量）后的大小是32*16*16
    //  kTileM = 128 KTileN = 128 kTileK = 32
    //  则有MMA_M=kTileM/TiledMMA_M=128/32=8 以此类推
    
    // 累积区域填充0
    clear(tCrD);

    // TiledCopy ldmatrix（S->R）
    // 直接利用tiled_mma的信息 形成块状拷贝 原因是tiledMMA包含了计算所需要的数据描述
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    // S和D分别表明是source和destination
    Tensor tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
    // 为了符合ldmatrix的layout，因此对tCrA进行retile，tCrA在调用partition_fragment_A时已经进行排布
    Tensor tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    Tensor tBsB = s2r_thr_copy_b.partition_S(sB);      // (CPY, CPY_M, CPY_K, kStage)
    Tensor tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_M, CPY_K)

    // TiledCopy (G->S)
    G2SCopyA g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    Tensor tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    Tensor tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

    G2SCopyB g2s_tiled_copy_b;
    ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    Tensor tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
    Tensor tBsB_copy = g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

    int itile_to_read = 0; //
    int ismem_read = 0;    // 
    int ismem_write = 0;   //
    
    // 提交 kStage - 1 tile 的 Copy
    // gmem -> shm
#pragma unroll
    for (int istage = 0;  istage < kStage - 1; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
                   tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
                   tBsB_copy(_, _, _, istage));
        cp_async_fence();
        
        ++itile_to_read;
        ++ismem_write;
    }

    // 允许剩余(kStage-2)组异步拷贝任务未完成，即等待第(kStage-2)-(kStage-1)=1组copy执行完毕
    cp_async_wait<kStage - 2>(); 
    __syncthreads();

    int ik = 0;
    // 第一组G->S完成后紧接S->R
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    // loop over k: i. load tile
    //              ii. mma
    int ntile = k / kTileK;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);  // nk = 2

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            // shm -> reg s[itile][ik + 1] -> r[ik + 1]
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                        tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                        tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                            tAsA_copy(_, _, _, ismem_write));
                cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                            tBsB_copy(_, _, _, ismem_write));
                ++itile_to_read;
                ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }
    // 使用较少的共享内存作为一个临时缓存，以使用宽指令
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{}); // 此时A的shm空间已不再使用，因此分配给C
    
    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    Tensor tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
    Tensor tCsC_r2s = r2s_thr_copy_c.partition_D(sC);
    // (CPY, CPY_M, CPY_N) = (8, 4, 4)
    // (CPY, _1, _1, pipe) = (8, 1, 1, 2)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    Tensor tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
    Tensor tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

    // 将[1,3)的主轴层级化为一个新的层级 (CPY, CPY_M, CPY_N) -> (CPY_, CPY_MN)
    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

    int step = size<3>(tCsC_r2s); // pipe
#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        // reg -> shm
#pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();
#pragma unroll
        // shm -> global
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    }

}

namespace config {

using namespace cute;

template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 5, int kSmemLayoutCBatch_ = 2, typename ComputeType = T_>
// kStage_意味着流水线阶段数，KSmemLayoutCBatch_控制共享内存中批处理布局的策略
struct GemmConfig {
    using T = T_;

    // Tile 配置
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kStage = kStage_;
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

    // Swizzle 重排share memory布局
    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 3;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtom{},
                      make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtom{},
                      make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

    // TileMMA 配置
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    
    using MMA = TiledMMA<
        mma_atom,
        Layout<Shape<Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>>>,
        Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>
        >

    // TileCopy 配置

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>; // 指示了数据只在L2做Cache，对L1则做bypass
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    
    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                                              make_layout(make_shape(Int<32>(), Int<4>{}),
                                                          make_stride(Int<4>{}, Int<1>{})),
                                              make_layout(make_shape(Int<1>{}, Int<8>{}))));
                                              // 和make_tiled_mma类似，制定线程和数据的重复方法将copy_atom拓展至块状。
    using G2SCopyB = G2SCopyA;

    // ldmatrix 配置

    using s2r_copy_op = SM75_U32x4_LDSM_N // 选择ldmatrix指令的x4模式。形成Atom抽象
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // register 途径 share memory 到 global memory
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                        make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{}))); // LayoutAtom拓展到指定大小的Layout

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                 "C shared memory request is large than A's one pipe")

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(S2GCopyAtomC{},
                                              make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                          make_stride(Int<4>{}, Int<1>{})),
                                              make_layout(make_shape(Int<1>{}, Int<8>{}))));
    static constexpr int kThreadNum = size(MMA{});
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

    static constexpr int kShmSize =
        cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}

int main(int argc, char *argv[]) {
    using T = cute::half_t;
    using namespace cute;
    using X = Underscore;

    srand(10086);

    cublasHandle_t handle;
    cublasCreate(&handle);
    int cublas_version;
    cublasGetVersion_v2(handle, &cublas_version);
    printf("cuBLAS version: %d\n", cublas_version);

    // default
    int M = 81920;
    int N = 256;
    int K = 256;
    
    int enable_cpu = 0;
    int enable_cublaslt = 1;
    int nt = 11;

    using ComputeType = T;

    T *Aptr;
    T *Bptr;
    T *Dptr;
    T *Dptr_cublas;
    T *Dptr_cublaslt;

    T *Aptr_host;
    T *Bptr_host;
    T *Dptr_host;
    T *Dptr_host_cpu;
    T *Dptr_host_blas;
    T *Dptr_host_cublaslt;

    Aptr_host = (T *)malloc(sizeof(T) * M * K);
    Bptr_host = (T *)malloc(sizeof(T) * N * K);
    Dptr_host = (T *)malloc(sizeof(T) * M * N);

    Dptr_host_cpu = (T *)malloc(sizeof(T) * M * N);
    Dptr_host_blas = (T *)malloc(sizeof(T) * M * N);
    Dptr_host_cublaslt = (T *)malloc(sizeof(T) * M * N);

    cudaMalloc(&Aptr, sizeof(T) * M * K);
    cudaMalloc(&Bptr, sizeof(T) * N * K);
    cudaMalloc(&Dptr, sizeof(T) * M * N);
    cudaMalloc(&Dptr_cublas, sizeof(T) * M * N);
    cudaMalloc(&Dptr_cublaslt, sizeof(T) * M * N);

    auto tA = make_tensor(Aptr_host, make_shape(M, K), make_stride(K, 1));
    auto tB = make_tensor(Bptr_host, make_shape(N, K), make_stride(K, 1));
    auto tD = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));

    cpu_rand_data(&tA);
    cpu_rand_data(&tB);

    clear(tD);

    cudaMemcpy(Aptr, Aptr_host, sizeof(T) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(Bptr, Bptr_host, sizeof(T) * N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(Dptr, Dptr_host, sizeof(T) * M * N, cudaMemcpyHostToDevice);
    cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
    cudaMemset(Dptr_cublaslt, 0, sizeof(T) * M * N);

    CublasLtGemm<T, ComputeType> cublaslt_gemm;
    if (enable_cublaslt) {
        cublaslt_gemm.init(Dptr_cublaslt, Bptr, Aptr, N, M, K);
    }

  config::GemmConfig<T, 128, 128, 32, 3> gemm_config;

  print(typename decltype(gemm_config)::MMA{});

  dim3 block = gemm_config.kThreadNum;
  dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
            (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
  int shm_size = gemm_config.kShmSize;

  half alpha = 1.f;
  half beta = 0.f;

  for (int it = 0; it < nt; ++it) {
    // blas
    cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                                     &alpha, (half *)Bptr, K, (half *)Aptr, K,
                                     &beta, (half *)Dptr_cublas, N);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("cublas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }

    if (enable_cublaslt) {
      cudaMemset(Dptr_cublaslt, 0, sizeof(T) * M * N);
      cublaslt_gemm.run();
    }

    // multi-stage
    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    gemm_multi_stage<decltype(gemm_config)>
        <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaMemcpy(Dptr_host, Dptr, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(Dptr_host_blas, Dptr_cublas, sizeof(T) * M * N,
                cudaMemcpyDeviceToHost);
    cudaMemcpy(Dptr_host_cublaslt, Dptr_cublaslt, sizeof(T) * M * N,
                cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("block = (%d, %d), gird = (%d, %d), shm = %d\n", block.x, block.y,
            grid.x, grid.y, shm_size);

    if (err == cudaSuccess) {
        printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    } else {
        printf_fail("err = %d, str = %s\n", err, cudaGetErrorString(err));
    }

    gpu_compare(Dptr, Dptr_cublas, M * N);

    if (enable_cublaslt) {
        gpu_compare(Dptr, Dptr_cublaslt, M * N);
    }

    auto tD_host = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));
    auto tD_host_cpu =
        make_tensor(Dptr_host_cpu, make_shape(M, N), make_stride(N, 1));
    auto tD_host_blas =
        make_tensor(Dptr_host_blas, make_shape(M, N), make_stride(N, 1));
    auto tD_host_cublaslt =
        make_tensor(Dptr_host_cublaslt, make_shape(M, N), make_stride(N, 1));

    if (enable_cpu) {
        cpu_gemm(&tD_host_cpu, tA, tB);
        cpu_compare(tD_host_cpu, tD_host, 0.1f);
    }

    auto tile = make_tile(min(8, M), min(8, N));
    auto t32x32 = local_tile(tD_host, tile, make_coord(0, 0));
    auto t32x32_cpu = local_tile(tD_host_cpu, tile, make_coord(0, 0));
    auto t32x32_blas = local_tile(tD_host_blas, tile, make_coord(0, 0));
    auto t32x32_cublaslt = local_tile(tD_host_cublaslt, tile, make_coord(0, 0));

    printf("M = %d, N = %d, K = %d\n", M, N, K);

    printf("our-impl:\n");
    print_tensor(t32x32);
    if (enable_cpu) {
        printf("cpu:\n");
        print_tensor(t32x32_cpu);
    }
    printf("cublas:\n");
    print_tensor(t32x32_blas);

    if (enable_cublaslt) {
        printf("cublaslt:\n");
        print_tensor(t32x32_cublaslt);
    }
}
