#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cooperative_groups.h>
#define TYPE int

using namespace cooperative_groups;

__global__ void my_kernel(int* a){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	a[tid]=0;
}

int main(int argc, char **argv){
	int dev = 1;
	int numBlocksPerSm = 0;
	int numThreads = 1024;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	std::cout << "BS SM BpSM TB TTHREADS \"SMs\""<<std::endl;	
	for (int i = 32; i < 1025; i +=8){
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, i, 1024);
		std::cout << i << ": ";
		std::cout << deviceProp.multiProcessorCount <<" * "<< numBlocksPerSm <<" = " << deviceProp.multiProcessorCount * numBlocksPerSm;
		std::cout << " " << deviceProp.multiProcessorCount * numBlocksPerSm * i;	
		std::cout << " " << (deviceProp.multiProcessorCount * numBlocksPerSm * i)/1024 << std::endl;
		//std::cout << i << "," << deviceProp.multiProcessorCount * numBlocksPerSm * i << std::endl;
	}
}
