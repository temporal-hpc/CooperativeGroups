#include <stdio.h>
#include <cuda.h>
#include <cooperative_groups.h>
#define TYPE int

using namespace cooperative_groups;

__device__ inline int h(TYPE a, TYPE b, TYPE c){
	return (a+b+c) & 0x1;
}
__global__ void ac(TYPE *vec1, TYPE *vec2, int n){
	int tid = blockDim.x * blockIdx.x + threadIdx.x+1;

	if(tid < n + 1 )
		vec2[tid] = h(vec1[(tid-1)], vec1[tid], vec1[(tid+1)]);
		
}
__global__ void ac_cg(TYPE *vec1, TYPE *vec2, int n, int REPEATS, int cpt){
	int tid = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int gridSize = gridDim.x * blockDim.x;

	grid_group grid = this_grid();
	int ntid;

	for(int i = 0; i < REPEATS; i++){


		// Vec2  <<<<< Vec1
		ntid = tid;
		for(int j = 0; j < cpt; j++){
			vec2[ntid] = h(vec1[ntid - 1], vec1[ntid], vec1[ntid + 1]);
			ntid += gridSize;
		}	
		if(ntid < n + 1 )
			vec2[ntid] = h(vec1[ntid - 1], vec1[ntid], vec1[ntid + 1]);
		sync(grid);

		// Vec1 <<<<<< Vec2
		ntid = tid;
		for(int j = 0; j < cpt; j++){
			vec1[ntid] = h(vec2[ntid - 1], vec2[ntid], vec2[ntid + 1]);
			ntid += gridSize;
		}
		if(ntid < n + 1 )
			vec1[ntid] = h(vec2[ntid - 1], vec2[ntid], vec2[ntid + 1]);
		sync(grid);
	}
}

__global__ void ac_sm(TYPE *vec1, TYPE *vec2, int n){
	int tid = blockDim.x * blockIdx.x + threadIdx.x+1;
	int lid = threadIdx.x + 1;

	__shared__ TYPE sm[BSIZE1D+2];
	if(tid < n + 1 ){
		//load shared memory
		sm[lid] = vec1[tid];
		if (lid == 1)	sm[lid-1] = vec1[tid-1];
		if (lid == BSIZE1D || tid == n)	sm[lid+1] = vec1[tid+1];
		__syncthreads();

		vec2[tid] = h(sm[(lid-1)], sm[lid], sm[(lid+1)]);
	}
		
}
__global__ void ac_cg_sm(TYPE *vec1, TYPE *vec2, int n, int REPEATS, int cpt){
	int tid = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int lid = threadIdx.x + 1;
	int gridSize = gridDim.x * blockDim.x;
	__shared__ TYPE sm[ (BSIZE1D+2) ];

	grid_group grid = this_grid();
	int ntid;

	for(int i = 0; i < REPEATS; i++){


		// Vec2  <<<<< Vec1
		ntid = tid;
		for(int j = 0; j < cpt; j++){
			//load Shared Memory
			sm[lid]=vec1[ntid];
			if (lid == 1)	sm[lid-1] = vec1[ntid-1];
			if (lid == BSIZE1D)	sm[lid+1] = vec1[ntid+1];
			__syncthreads();

			vec2[ntid] = h(sm[lid - 1], sm[lid], sm[lid + 1]);
			ntid += gridSize;
		}	

		//load Shared Memory
		if(ntid < n + 1 ){
			sm[lid]=vec1[ntid];
			if (lid == 1)	sm[lid-1] = vec1[ntid-1];
			if (ntid == n || lid == BSIZE1D )	sm[lid+1] = vec1[ntid+1];
		}
		__syncthreads();

		if(ntid < n + 1 ){
			vec2[ntid] = h(sm[lid - 1], sm[lid], sm[lid + 1]);
		}
		sync(grid);

		// Vec1 <<<<<< Vec2
		ntid = tid;
		for(int j = 0; j < cpt; j++){
			//load Shared Memory
			sm[lid]=vec2[ntid];
			if (lid == 1)	sm[lid-1] = vec2[ntid-1];
			if (lid == BSIZE1D)	sm[lid+1] = vec2[ntid+1];
			__syncthreads();

			vec1[ntid] = h(sm[lid - 1], sm[lid], sm[lid + 1]);
			ntid += gridSize;
		}	

		//load Shared Memory
		if(ntid < n + 1 ){
			sm[lid]=vec2[ntid];
			if (lid == 1)	sm[lid-1] = vec2[ntid-1];
			if (ntid == n || lid == BSIZE1D )	sm[lid+1] = vec2[ntid+1];
		}
		__syncthreads();

		if(ntid < n + 1 ){
			vec1[ntid] = h(sm[lid - 1], sm[lid], sm[lid + 1]);
		}
		sync(grid);
	}
}

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
__host__ void compare(TYPE *a, TYPE *b, int n, int option, const char *s){
	if(option == 1){
		for (int i = 0; i < n; i++){
			if(a[i]!=b[i])	printf("%3i ", i);
		}
		printf("\n");
	}
	if(option == 0){
		int i = 0;
		while(i < n && a[i]==b[i])	i++;
		printf("%s %s , %i\n", s,(n==i) ? "OK" : "FAIL", i);
	}
}
__host__ void last_cuda_error(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("[%s]: CUDA error: %s\n", msg, cudaGetErrorString(error));
		exit(-1);
	}
}
__host__ void printVec(int *a, int n, int limit){
	for(int i = 0; i < ( n < limit ? n : limit); i++)
		printf("%d", a[i]);
	printf("\n");
}

int main(int argc, char **argv){
	if(argc != 5){
		printf("Error!, Ejecutar ./prog <N> <REPEATS> <seed>\n");
		exit(1);
	}
	int n = atoi(argv[1]);
	int REPEATS = atoi(argv[2]);
	int seed = atoi(argv[3]);
	int dev = atoi(argv[4]);
	seed = seed==0 ? time(NULL) : seed;
	srand(seed);
	cudaSetDevice(dev);
	
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	
	printf("seed = %i\n",seed);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;
	TYPE *a, *b,/* *s,*/ *a_d, *b_d/*, *s_d*/, *final_, *final_cg, *final_sm, *final_cg_sm;

	a =(TYPE*)malloc(sizeof(TYPE)*(n+2));
	b =(TYPE*)malloc(sizeof(TYPE)*(n+2));
	final_ =(TYPE*)malloc(sizeof(TYPE)*(n+2));
	final_sm =(TYPE*)malloc(sizeof(TYPE)*(n+2));
	final_cg =(TYPE*)malloc(sizeof(TYPE)*(n+2));
	final_cg_sm =(TYPE*)malloc(sizeof(TYPE)*(n+2));
//	s =(TYPE*)malloc(sizeof(TYPE)*n);

	cudaMalloc(&a_d, sizeof(TYPE)*(n+2));
	cudaMalloc(&b_d, sizeof(TYPE)*(n+2));
//	cudaMalloc(&s_d, sizeof(TYPE)*n);
	
	fillVector(a, n+2);
	copyVec(a, final_cg, n+2);
	copyVec(a, final_sm, n+2);
	copyVec(a, final_cg_sm, n+2);

//	fillVector(b, n);
/*
	dim3 block(BSIZE2D, BSIZE2D, 1);
	dim3 grid( (n + block.x - 1)/block.x, (n + block.y - 1)/block.y );
*/	
	dim3 block(BSIZE1D, 1, 1);
	dim3 grid( ((n) + block.x - 1)/block.x, 1, 1);

//	printf("block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);

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

	cudaMemcpy(final_, a_d, sizeof(TYPE) * (n+2), cudaMemcpyDeviceToHost);
#ifndef DEBUG
	printf("GPU Simple...ok! in %f ms\n", ms);



	// Simple con Shared Memory
	cudaMemcpy(a_d, final_sm, sizeof(TYPE) * (n+2), cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	for(int i = 0; i< REPEATS; i++){
		ac_sm<<<grid,block>>>(a_d, b_d, n);
		cudaDeviceSynchronize();
		ac_sm<<<grid,block>>>(b_d, a_d, n);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	last_cuda_error("ERROR AC_SM:");
	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	printf("GPU Simple_SM...ok! in %f ms\n", ms);


	cudaMemcpy(final_sm, a_d, sizeof(TYPE) * (n+2), cudaMemcpyDeviceToHost);

	//Cooperative groups

	int numBlocksPerSm = 0;
	int maxNumThreads = 0;
	int cellPerThread = 1;

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, ac_cg, BSIZE1D, 0);
	maxNumThreads = numBlocksPerSm * deviceProp.multiProcessorCount * BSIZE1D;
//	printf("%i\n", cellPerThread);

	if(maxNumThreads < n){
		cellPerThread = (n + maxNumThreads - 1) / maxNumThreads;
//		printf("%i\n", cellPerThread);
		grid = dim3( ( ( (n + cellPerThread - 1) / cellPerThread) + block.x - 1) /block.x, 1, 1 );
//		printf("block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
	}

	cellPerThread--;

	// Sin Shared Memory
	cudaMemcpy(a_d, final_cg, sizeof(TYPE) * (n+2), cudaMemcpyHostToDevice);

    	void* params[5];

    	params[0] = (void *)&a_d;  
    	params[1] = (void *)&b_d;
    	params[2] = (void *)&n;
    	params[3] = (void *)&REPEATS;
    	params[4] = (void *)&cellPerThread;

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


	// Con Shared Memory
	cudaMemcpy(a_d, final_cg_sm, sizeof(TYPE) * (n+2), cudaMemcpyHostToDevice);

    	params[0] = (void *)&a_d;  
    	params[1] = (void *)&b_d;
    	params[2] = (void *)&n;
    	params[3] = (void *)&REPEATS;
    	params[4] = (void *)&cellPerThread;

	cudaEventRecord(start);

    	cudaLaunchCooperativeKernel((void *) ac_cg_sm, grid, block, params, 0, NULL);
    	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	last_cuda_error("ERROR COOPERATIVE GROUPS SM:");

	cudaMemcpy(final_cg_sm, a_d, sizeof(TYPE) * n, cudaMemcpyDeviceToHost);
	printf("GPU CG_SM...ok! in %f ms\n", ms);


	printf("\n");
	compare(final_, final_sm, n, 0, "Simple_SM");
	compare(final_, final_cg, n, 0, "CG       ");
	compare(final_, final_cg_sm, n, 0, "CG_SM    ");

	printVec(final_, n, 100);
	printVec(final_cg, n, 100);
	printVec(final_sm, n, 100);
	printVec(final_cg_sm, n, 100);
#endif
	return 0;
}
