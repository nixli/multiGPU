#include <sys/socket.h> 
#include <stdlib.h> 
#include <netinet/in.h> 
#include <arpa/inet.h>
#include <string.h> 
#include <iostream>
#include <cstdlib>
#include <math.h>
#include "stdio.h"
#include <assert.h>
#include <iomanip>
#include <sys/ipc.h> 
#include <sys/shm.h> 
#include <sys/mman.h>
#include <sys/stat.h>        
#include <unistd.h>   /* For open(), creat() */
#include <fcntl.h>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "helper_cuda.h"
#include <thread>
#include "ipc.h"
#include "matrix.h"
#include <sys/mman.h>
#include <sstream>
#include "utils.h"
#define MAX_SMH_NAME_LEN
using namespace std;
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{   
    for (unsigned int i = 0; i < hA; ++i)
      for (unsigned int j = 0; j < wB; ++j)
        {   
            double sum = 0;
            
            for (unsigned int k = 0; k < wA; ++k)
            {   
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            
            C[i * wB + j] = (float)sum;
        }
}


#undef VERIFY



float* create_shmm(int id, int &shm_fd){

  std::stringstream ss;
  ss << "/gpu_" << id;
  const char* shm_name = ss.str().c_str();
  shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0644);
  if (shm_fd <0){
    std::cout << "Error, failed to create shared memory, error: " << strerror(errno) <<" \n";
    exit(-1);
  }

//  int shmid = shmget(key, X*Z*sizeof(float), 0660 | IPC_CREAT); 
//  if(shmid < 0){
//    std::cout << "Error, failed to open shared momory, ErrorNO: " << errno << std::endl;
//    exit(-1);
//  }
  const size_t size = X*Z*sizeof(float); 
  int rc = ftruncate(shm_fd, size);
  float *data = (float*) mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (data == (void*)-1){
    std::cout << "Failed to map memory to the current address space\n";
    exit(-1);
  }
  DEBUG_PRINT(std::cout << "mapped shared memory " << std::endl);
  return data;

}
int launch_kernel(int identifier, Matrix &A, Matrix &B, Matrix &C){

  // initialize random matrix
  DEBUG_PRINT(std::cout<<"matrix generated, ready to work on the device\n");
  A.cuda_malloc();
  B.cuda_malloc();
  C.cuda_malloc();
  DEBUG_PRINT(std::cout<<"starting to work on the device side of things\n");
  auto start_time = NOW();
  // coopy memory to cuda device
  A.toCUDA();
  B.toCUDA();
  auto end = NOW();
  DEBUG_PRINT(std::cout<<"time for copying data to device: " << DIFF_T(start_time, end) << " milliseconds" << std::endl);

  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cudaEvent_t stop, start;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));
  checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,\
                   X, Y, Z, &alpha, \
                   A.data_cuda, Y, B.data_cuda, Y, &beta, C.data_cuda, X));
  // since we are calling stop after the blas call, just call event synchronize
  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));

  float device_mil = 0;
  cudaEventElapsedTime(&device_mil, start, stop); 
  DEBUG_PRINT(std::cout<<"time taken on device: " << device_mil << std::endl);

  DEBUG_PRINT(std::cout << "creating shmm and copying from the device" << std::endl;) 
  start_time = NOW();
  int shm_fd;
  float* data = create_shmm(identifier, shm_fd);
  // float *data = (float*) shmat(shmid,(void*)0,0); 
  // copy device memory to shared memory
  C.fromCUDA(data);

  end = NOW();
  DEBUG_PRINT(std::cout<<"GPU is done, copying to shm took " << DIFF_T(start_time, end) << " milliseconds" << std::endl);
#ifdef _DEBUG
  matrixMulCPU(C.data_cpu, A.data_cpu, B.data_cpu, X, Y, Z);
  std::cout << "CPU IS DONE ! " << std::endl;
  std::cout << "ret val is: " <<cudaGetLastError() << " and the answer is: " << C.verify() << std::endl;;

  
          for(int j =0; j < X; ++j){
            for (int k =0; k < Z; ++k)
              std::cout << data[j*Z + k] <<" ";
            std::cout << std::endl;
          }
          for(int j =0; j < X; ++j){
            for (int k =0; k < Y; ++k)
              std::cout << A.data_cpu[j*Z + k] <<" ";
            std::cout << std::endl;
          }
          std::cout << "\n\n" <<std::endl;
          for(int j =0; j < Y; ++j){
            for (int k =0; k < Z; ++k)
              std::cout << B.data_cpu[j*Z + k] <<" ";
            std::cout << std::endl;
          }
#endif
  munmap(data,  X*Z*sizeof(float));
  close(shm_fd);
  // shmdt(data);

  A.freeDeviceMem();
  B.freeDeviceMem();
  C.freeDeviceMem();
  C.freeHostMem();
  return 0;
}


int connect_to_master(){
    int sockfd = 0; 
    struct sockaddr_in serv_addr; 

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return -1; 
    } 
   
    memset(&serv_addr, 0, sizeof(serv_addr)); 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(PORT); 
       
    // Convert IPv4 and IPv6 addresses from text to binary form 
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return -1; 
    } 
   
    if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) 
    { 
        printf("\nConnecntion Failed \n"); 
        return -1; 
    } 

    IPCCommand request = NODE_CONNECT;
    send(sockfd, &request, sizeof(IPCCommand) , 0 ); 
    std::cout << "First Request message sent\n";

    IPCCommand response;
    read( sockfd , &response, sizeof(IPCCommand));
    if(response != MASTER_ACK){
      return -1;
    }

    std::cout << "Connection accepted for GPU, socket fd is: " << sockfd<< " \n"; 
    return sockfd; 
}

void main_event_loop(int socketfd) {

    Matrix A(X, Y, false);
    Matrix B(Y, Z, false);
    Matrix C(X,Z);
    data_block_t from_master, to_master;
    int val_read = read(socketfd, &from_master, sizeof(data_block_t));
    while (val_read >0 && from_master.cmd != MASTER_NODE_SHUTDOWN){
      if (from_master.cmd == MASTER_INPUT_AVAILABLE) {
        
        auto start = NOW();
	// this launches a call to CUBLAS      
       	launch_kernel( from_master.identifier, A, B, C);
        auto end = NOW();

        DEBUG_PRINT(std::cout << "worker node one set of computation takes: " << DIFF_T(start, end) << "seconds\n\n" << std::endl);
	to_master.cmd = NODE_OUTPUT_AVAILABLE;
        to_master.identifier = from_master.identifier;

        send(socketfd, &to_master, sizeof(data_block_t),0);
        val_read = read(socketfd, &from_master, sizeof(data_block_t));
      }
   }
}


int send_fin(int sockfd){
    std::string finish = "Finished";
    send(sockfd, finish.c_str() , strlen(finish.c_str()) , 0 ); 
    printf("Finished!\n"); 
    char buffer[1024] = {0};
    int valread = read( sockfd , buffer, 1024); 
    printf("%s\n",buffer );
    return 0;
}


int main(int argc, char* argv[]){
    if (argc < 2){
      printf("Please provide a GPU device\n");
      return 1;
    }
    if (atoi(argv[1]) < 0){
      printf("Invalid argument\n");
      return 1;
    }
    int device = atoi(argv[1]);
    cudaSetDevice(device);
    int node_socket = connect_to_master();
    main_event_loop(node_socket);
    send_fin(node_socket);
    return 0;
}
