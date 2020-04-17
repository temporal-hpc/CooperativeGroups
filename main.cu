#include <stdio.h>
#include <cuda.h>
#include <cooperative_groups.h>
#define TYPE int

using namespace cooperative_groups;

__device__ inline int h(TYPE a, TYPE b, TYPE c){
	return (a+b+c) & 0x1;
}
__global__ void kernel(TYPE *s, TYPE *a, TYPE *b, int n){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//if(tid < n ){
	s[tid] += a[tid] * b[tid];
	//}
}
__global__ void kernel_cg(TYPE *s, TYPE *a, TYPE *b, int n, int REPEATS){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = 0; i < REPEATS; i++){
		//if(tid < n ){
		s[tid] += a[tid] * b[tid];
		//}
		sync(this_grid());
	}
}
__global__ void ac(TYPE *vec1, TYPE *vec2, int n){
	int tid = blockDim.x * blockIdx.x + threadIdx.x+1;

	vec2[tid] = h(vec1[(tid-1)], vec1[tid], vec1[(tid+1)]);
		
}
__global__ void ac_cg(TYPE *vec1, TYPE *vec2, int n, int REPEATS){
	int tid = blockDim.x * blockIdx.x + threadIdx.x+1;
	grid_group grid = this_grid();

	for(int i = 0; i < REPEATS; i++){
		vec2[tid] = h(vec1[(tid-1)], vec1[tid], vec1[(tid+1)]);
		sync(grid);

		vec1[tid] = h(vec2[(tid-1)], vec2[tid], vec2[(tid+1)]);

		sync(grid);
	}
}

/*
__host__ void fillMatrix(TYPE *a, int n){
	for(int i = 0; i < n * n; i++){
		a[i] = rand()%3-1;
	}
}*/

__host__ void fillVector(TYPE *a, int n){
	for(int i = 0; i < n; i++){
		a[i] = rand()%2;
	}
	a[0]=0;
	a[n]=0;
}
__host__ void copyVec(TYPE *a, TYPE *b, int n){	
	for(int i = 0; i < n; i++){
		b[i] = a[i];
	}
}
__host__ void fillZero(TYPE *a, int n){
	for(int i = 0; i < n; i++){
		a[i] = 0;//rand()%3-1;
	}
}
__host__ void compare(TYPE *a, TYPE *b, int n){
	for (int i = 0; i < n; i++){
		if(a[i]!=b[i])	printf("%3i ", i);
	}
	printf("\n");
	/*
	int i = 0;
	while(i < n && a[i]==b[i])	i++;
	printf("%s, %i\n", (n==i) ? "OK" : "FAIL", i);
	*/
}
__host__ void last_cuda_error(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("[%s]: CUDA error: %s\n", msg, cudaGetErrorString(error));
		exit(-1);
	}
}
__host__ void printVec(int *a, int n){
	for(int i = 0; i < 50; i++)
		printf("%d", a[i]);
	printf("\n");
}

int main(int argc, char **argv){
	if(argc != 4){
		printf("Error!, Ejecutar ./prog <N> <REPEATS> <seed>\n");
		exit(1);
	}
	int n = atoi(argv[1]);
	int REPEATS = atoi(argv[2]);
	int seed = atoi(argv[3]);
	seed = seed==0 ? time(NULL) : seed;
	srand(seed);
	cudaSetDevice(1);
	printf("seed = %i\n",seed);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;
	TYPE *a, *b,/* *s,*/ *a_d, *b_d/*, *s_d*/, *final_, *final_cg;

	a =(TYPE*)malloc(sizeof(TYPE)*(n+2));
	b =(TYPE*)malloc(sizeof(TYPE)*(n+2));
	final_ =(TYPE*)malloc(sizeof(TYPE)*(n+2));
	final_cg =(TYPE*)malloc(sizeof(TYPE)*(n+2));
//	s =(TYPE*)malloc(sizeof(TYPE)*n);

	cudaMalloc(&a_d, sizeof(TYPE)*(n+2));
	cudaMalloc(&b_d, sizeof(TYPE)*(n+2));
//	cudaMalloc(&s_d, sizeof(TYPE)*n);
	
	fillVector(a, n+2);
	copyVec(a, final_cg, n+2);
//	fillVector(b, n);
/*
	dim3 block(BSIZE2D, BSIZE2D, 1);
	dim3 grid( (n + block.x - 1)/block.x, (n + block.y - 1)/block.y );
*/	
	dim3 block(BSIZE1D, 1, 1);
	dim3 grid( ((n) + block.x - 1)/block.x, 1, 1);

	printf("block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);

	cudaMemcpy(a_d, a, sizeof(TYPE) * (n+2), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, sizeof(TYPE) * (n+2), cudaMemcpyHostToDevice);
//	cudaMemcpy(s_d, a, sizeof(TYPE) * n, cudaMemcpyHostToDevice);

#ifdef DEBUG
	printVec(a, n);
	getchar();
#endif
	//Flujo normal
	cudaEventRecord(start);

	for(int i = 0; i< REPEATS; i++){
/*
		kernel<<<grid,block>>>(s_d, a_d, b_d, n);
		cudaDeviceSynchronize();
*/
		ac<<<grid,block>>>(a_d, b_d, n);
		cudaDeviceSynchronize();
		#ifdef DEBUG
			cudaMemcpy(b, b_d, sizeof(TYPE) * (n+2), cudaMemcpyDeviceToHost);
			printVec(b, n);
			getchar();
		#endif
		ac<<<grid,block>>>(b_d, a_d, n);
		cudaDeviceSynchronize();
		#ifdef DEBUG		
			cudaMemcpy(a, a_d, sizeof(TYPE) * (n+2), cudaMemcpyDeviceToHost);
			printVec(a, n);
			getchar();
		#endif
	}
	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	last_cuda_error("ERROR AC:");
	
/*
	fillZero(s, n);
	cudaMemcpy(s_d, a, sizeof(TYPE) * n, cudaMemcpyHostToDevice);
*/
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

#ifndef DEBUG
	printf("GPU Simple...ok! in %f ms\n", ms);

	cudaMemcpy(final_, a_d, sizeof(TYPE) * (n+2), cudaMemcpyDeviceToHost);

	//Cooperative groups
	copyVec(final_cg, a, n+2);

//	printVec(a, n);
//	printVec(final_cg, n);
	cudaMemcpy(a_d, final_cg, sizeof(TYPE) * (n+2), cudaMemcpyHostToDevice);

    	void* params[4];

    	params[0] = (void *)&a_d;  
    	params[1] = (void *)&b_d;
    	params[2] = (void *)&n;
    	params[3] = (void *)&REPEATS;

	cudaEventRecord(start);

    	cudaLaunchCooperativeKernel((void *) ac_cg, grid, block, params, 0, NULL);
    	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	last_cuda_error("ERROR COOPERATIVE GROUPS:");

	cudaMemcpy(final_cg, a_d, sizeof(TYPE) * n, cudaMemcpyDeviceToHost);
	printf("GPU CG...ok! in %f ms\n", ms);

	compare(final_, final_cg, n);

	printVec(final_, n);
	printVec(final_cg, n);
#endif
	return 0;
}
