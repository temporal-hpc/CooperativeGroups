#include <stdio.h>
#include <cuda.h>
#include <cooperative_groups.h>
#define TYPE int

using namespace cooperative_groups;

__global__ void kernel(){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	

}

__host__ void fillMatrix(TYPE *a, int n){
	for(int i = 0; i < n * n; i++){
		a[i] = rand()%3-1;
	}
}

__host__ void fillVector(TYPE *a, int n){
	for(int i = 0; i < n; i++){
		a[i] = rand()%3-1;
	}
}

int main(int argc, char **argv){
	if(argc != 3){
		printf("Error!, Ejecutar ./prog <OPTION><N> <seed>\nOPTION:\n1 = VECTOR\n2 = MATRIX\n");
		exit(1);
	}
	int n = atoi(argv[1]);
	int seed = atoi(argv[2]);
	srand(seed);

	dim3 block(BSIZE2D, BSIZE2D, 1);
	dim3 grid( (n + block.x - 1)/block.x, (n + block.y - 1)/block.y );

	return 0;
}
