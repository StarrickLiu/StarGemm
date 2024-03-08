# StarGemm

实践与比较Cuda平台的高性能矩阵通用乘。

本仓库主要参考reed-lau老师的实现 https://github.com/reed-lau/cute-gemm

过了一遍实现的同时，在原实现的基础上添加了诸多注释，便于初学者理解

## 使用方法

```
git clone https://github.com/StarrickLiu/StarGemm.git
git submodule update
make
cd build
./gemm-starrick
```

## 未来计划

基于Cutlass实现并比较stream-k等方法在不同规模Gemm上的性能

## 推荐阅读

如何使用Cute实现高效gemm

https://zhuanlan.zhihu.com/p/675308830

Cutlass3.4 之后参数输入的变化

https://github.com/NVIDIA/cutlass/discussions/1345