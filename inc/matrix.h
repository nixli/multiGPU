#ifndef MATRIX_H
#define MATRIX_H

#include <string.h> 
#include <iostream>
#include <cstdlib>
#include <cmath>

struct Matrix {
  float *data_cpu;
  float *data_cuda;
  float *cuda_res;
  size_t row_dim;
  size_t col_dim;
  bool host_mem_shared;
  Matrix(size_t _x, size_t _y, bool rand_gen=false, float init=0){

    row_dim = _x;
    col_dim = _y;
    data_cpu = (float*)malloc(row_dim*col_dim*sizeof(float));
    if (rand_gen) srand(1023);
    for (int i =0; i< row_dim; ++i){
      for(int j =0; j < col_dim; ++j){
        data_cpu[i*col_dim + j] = rand_gen? rand() / (float)RAND_MAX :init;
      }
    }
    host_mem_shared = false;
  }
  void cuda_malloc(){
    assert(cudaMalloc((float**)&data_cuda, row_dim*col_dim*sizeof(float)) == cudaSuccess);
  }
  // transfers data to the device, also allocates memoey
  void toCUDA(){
    cudaError_t result = cudaMemcpy(data_cuda, data_cpu, row_dim*col_dim*sizeof(float), cudaMemcpyHostToDevice);
    assert(result==cudaSuccess);
  }
  // transfers data from the device, assume a values are freed
  void fromCUDA(float * hostmem){
    if(hostmem == NULL)
      assert(cuda_res = (float*)malloc(row_dim * col_dim*sizeof(float)));
    else{
      host_mem_shared = true;
      cuda_res = hostmem;
    }

    cudaError_t result = cudaMemcpy(cuda_res, data_cuda, row_dim*col_dim*sizeof(float), cudaMemcpyDeviceToHost);
    assert(result==cudaSuccess);
  }
  void freeHostMem(){
    if(!host_mem_shared)
      free(cuda_res);
  }
  void freeDeviceMem(){
    cudaFree(data_cuda);
  }
  void printGPU(){
    std::cout << "MATRIX: " << std::endl;
    for (int i =0; i<row_dim; ++i){
      for(int j =0; j < col_dim; ++j){
        std::cout << std::setw(10) << cuda_res[col_dim*i + j] << " ";
      }
      std::cout <<std::endl;
    }

  }
  void printCPU(){
    std::cout << "MATRIX: " << std::endl;
    for (int i =0; i<row_dim; ++i){
      for(int j =0; j < col_dim; ++j){
        std::cout << std::setw(10) << data_cpu[col_dim*i + j] << " ";
      }
      std::cout <<std::endl;
    }

  }
  bool verify(){
    bool status = true;
    std::cout << "\nVerifying results" <<std::endl;
    for (int i =0; i<row_dim; ++i){
      for(int j =0; j < col_dim; ++j){
        if(std::abs(data_cpu[col_dim*i + j] - cuda_res[col_dim*i + j]) > 0.01) {
          std::cout << "at [" <<i<< " " <<j <<"], "\
            << "CPU is: " << data_cpu[col_dim*i + j] << " but GPU is: " << cuda_res[col_dim*i + j] <<std::endl;
          status =  false;
        }
      }
    }
    return status;
  }
};

#endif
