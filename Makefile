DEBUG=NONE
BSIZE1D=32
BSIZE2D=32
ARCH=sm_70
PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -arch ${ARCH} --std=c++11 -default-stream per-thread -rdc=true
SRC=main.cu
BINARY=bin/prog
all:
	nvcc ${PARAMS} ${SRC} -o ${BINARY}

test:
	nvcc ${PARAMS} test.cu -o bin/test
