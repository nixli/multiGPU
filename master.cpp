#include <unistd.h> 
#include <stdio.h> 
#include <sys/socket.h> 
#include <stdlib.h> 
#include <netinet/in.h> 
#include <string.h> 
#include <iostream>

#include <sys/ipc.h> 
#include <sys/shm.h> 
#include "ipc.h"

#define X 1000
#define Y 1000
#define Z 1000
#undef VERIFY

int nodes[NUMGPU] = {0};
static int num_nodes_connected = 0;

typedef struct sockaddr_in sockaddr_in;

sockaddr_in* create_master_server(int &masterfd) 
{ 
    std::cout << "creating server" <<std::endl;
    int valread; 
    sockaddr_in *address = new sockaddr_in;
    int opt = 1; 
       
    // Creating socket file descriptor 
    if ((masterfd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
       
    if (setsockopt(masterfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) 
    { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 
    address->sin_family = AF_INET; 
    address->sin_addr.s_addr = INADDR_ANY; 
    address->sin_port = htons( PORT ); 
       
    // Forcefully attaching socket to the port 8080 
    if (bind(masterfd, (struct sockaddr *)address,  
                                 sizeof(sockaddr_in))<0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    }

    if (listen(masterfd, 3) < 0) 
    { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
  }
    std::cout <<"Done! Socket FD is " << masterfd << std::endl;
    return address; 
}


int accept_node(sockaddr_in *address, int masterfd){

    int new_socket, valread;
    int addrlen = sizeof(sockaddr_in); 
    if ((new_socket = accept(masterfd, (struct sockaddr *)address,  
                       (socklen_t*)&addrlen))<0) 
    { 
        perror("accept"); 
        exit(EXIT_FAILURE); 
    }

    IPCCommand command;
    valread = read( new_socket , &command, sizeof(command));
    if (command != NODE_CONNECT){
      return -1;
    } 

  
    nodes[num_nodes_connected] = new_socket;
    num_nodes_connected++;
    std::cout << "Connection Established for " << new_socket <<  " !\n";
    
    IPCCommand response = MASTER_ACK;
    send(new_socket , &response , sizeof(response) , 0 ); 
    return new_socket;
}

int test(){
    IPCCommand master_ipc = MASTER_INPUT_AVAILABLE;
    IPCCommand nodes_ipc[NUMGPU];
    for(int i =0; i < NUMGPU; ++i){
        send(nodes[i], &master_ipc, sizeof(IPCCommand), 0);
    }

    for(int i =0; i < NUMGPU; ++i){
        int val_read = read(nodes[i], nodes_ipc+i, sizeof(IPCCommand));

        if(val_read >0 && *(nodes_ipc+i)==NODE_OUTPUT_AVAILABLE){
          std::cout << "SUCCESS, GPU computation finished\n";
          key_t key = ftok("res",1234);            
          int shmid = shmget(key, X*Z*sizeof(float), 0660|IPC_CREAT); 
          float *data = (float*) shmat(shmid,(void*)0,0); 
//#define _DEBUG
#ifdef _DEBUG
          for(int j =0; j < X*Z; ++j){
            std::cout << data[j] <<" ";
          } 
#endif
          shmctl(shmid,IPC_RMID,NULL); 

        }
    }
}

int main(){
    int masterfd=0;
    sockaddr_in* sock = create_master_server(masterfd);
    std::cout << "waiting for " << NUMGPU << " devices to connect\n";
    for(int i =0; i < NUMGPU; ++i)
      accept_node(sock, masterfd);
    for (int i = 0; i < 10; ++i)
      test();
}
