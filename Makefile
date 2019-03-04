CC=g++
NVCC=nvcc
CFLAGS=-Wall
INC := ./inc

node: node.cpp ${inc}/*
	$(NVCC) node.cpp -std=c++11 -lcublas -I$(INC) -o node

master: master.cpp ${inc}/*
	$(CC) master.cpp -I$(INC) -o master

kernel: node.cu ${inc}/*
	$(NVCC) node.cu -std=c++11 -I$(INC) -o node_kernel -DGPU

all: node master kernel 
clean :
	$(RM) node master node_kernel
