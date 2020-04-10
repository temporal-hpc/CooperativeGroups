DEBUG=NONE
BSIZE1D=512
BSIZE2D=32
ARCH=sm_75
PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -arch ${ARCH} --std=c++11 -default-stream per-thread -rdc=true
SRC=main.cu
BINARY=bin/prog
all:
	nvcc ${PARAMS} ${SRC} -o ${BINARY}
