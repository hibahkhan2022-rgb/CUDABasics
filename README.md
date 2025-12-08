# CUDA Basics
This repo explains CUDA fundamentals and parallelization 

### GPU vs CPU
- CPU excels at sequence of operations, whereas GPU excels at executing operations in parallel (many transistors for data processing)
- A kernel is a function executed on the GPU, and is instantiated by adding the __global__identifier

```
//Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
```
### Streaming Multiprocessors
- Execution unit of the GPU that executes the warps
- Warp: A group of 32 threads; they execute the same instruction
- Grid: Collection of blocks, max number of blocks is dependent on the device
    For example: THe T4 GPU can run 9 quintillion blocks
- Block: Groups of threads on the GPU
- Thread: Smallest unit of execution in CUDA. Each thread executes the same code but operates on different data; all threads are guaranteed to be synchornized

One CUDA warp can be processed by one cuda core only. 

```
- gridDim: How many blocks in each x,y,z axis in the grid
- blockDim: How many threads in each x,y,z in the block
- blockIdx: x,y,z index of the block position in the grid
- threadIdx: x,y,z index of the thread position in the block



