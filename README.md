# CUDABasics
This repo explains CUDA fundamentals and parallelization 

##GPU vs CPU
- CPU excels at sequence of operations, whereas GPU excels at executing operations in parallel (many transistors for data processing)

A kernel is a function executed on the GPU, and is instantiated by adding the __global__identifierf

```
//Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}



