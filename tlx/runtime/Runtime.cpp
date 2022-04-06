
#include <cuda.h>
#include <cuda_runtime.h>


void AllocateDeviceMem(void **DevicePtr, uint64_t Size) {
  cudaMalloc(DevicePtr, Size);
}

void HostToDeviceCopy(void *HostPtr, void **DevicePtr, uint64_t Size) {
  cudaMalloc(DevicePtr, Size);
  cudaMemcpy(HostPtr, *DevicePtr, Size, cudaMemcpyHostToDevice);
}

void DeviceToHostCopy(void **DevicePtr, void *HostPtr, uint64_t Size) {
  cudaMemcpy(*DevicePtr, HostPtr, Size, cudaMemcpyDeviceToHost);
  cudaFree(*DevicePtr);
}

void KernelSyncLaunch(void *Func, void** Args, uint64_t GridDim, uint64_t BlockDim) {
  cudaLaunchKernel(Func, dim3(GridDim), dim3(BlockDim), Args);
  cudaDeviceSynchronize();
}


