CC=g++
NVCC=nvcc
INC := ./inc
STD=-std=c++11 
NUMGPU?=1
override CFLAGS += -lrt
node: node.cpp ${inc}/*
	$(NVCC) node.cpp $(STD)  $(CFLAGS) -lcublas -I$(INC)  -o node

master: master.cpp ${inc}/*
	$(CC) master.cpp -I$(INC) $(STD) $(CFLAGS)  -DNUMGPU=$(NUMGPU)  -o master

kernel: node.cu ${inc}/*
	$(NVCC) node.cu -std=c++11 -I$(INC) -o node_kernel -DGPU

all: node master kernel 
clean :
	$(RM) node master node_kernel
