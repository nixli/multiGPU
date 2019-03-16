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

#undef VERIFY

#define NUM_RESULTS_REQUIRED (NUMGPU/2+1)

int nodes[NUMGPU] = {0};
static int num_nodes_connected = 0;
static int max_fd = 0;

typedef struct sockaddr_in sockaddr_in;

struct compute_node_result {
  int node_id;
  float* data;
};

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
    if(new_socket > max_fd)
      max_fd = new_socket;
    std::cout << "Connection Established for " << new_socket <<  " !\n";
    
    IPCCommand response = MASTER_ACK;
    send(new_socket , &response , sizeof(response) , 0 ); 
    return new_socket;
}

int send_one_round_of_inputs(){

    fd_set select_fds, select_fds_main;
    for (int i =0; i < NUMGPU; ++i){
      FD_SET(nodes[i], &select_fds_main);
    }
    // select messes up with the set of FDs, we need to keep a clean copy
    select_fds = select_fds_main;

    // set the first set of inputs, we assume that data are visible across
    // all compute nodes, including the master and the worker. This assumption
    // is valid because the real data of a large matrix computation can be
    // 1. In memory of the reception node, in which case we could create a shared memory for
    // 2. Saved on the disk, in which case all worker nodes can read from
    
    // when we send a set of inputs (in this case the identifier of where to look for the data)
    // we need to keep track of which one we send since we need to re-build the results based on
    // which ones we have collected. The worker node, on the other hand, should also return the
    // identifier that was provided to it from this master
    for(int i =0; i < NUMGPU; ++i){
      data_block_t  request;
      request.cmd = MASTER_INPUT_AVAILABLE;
      // note that this i contains special meaning on how we encode the matrix
      request.identifier = i;
      // TODO: additionally, we need to provide a name of the shared memory for the worer node
      // to create and write the results to, here we are just using a dummy string
      send(nodes[i], &request, sizeof(request), 0);
    }

    int r_back =0;
    data_block_t  results_back[NUM_RESULTS_REQUIRED ];
 
    while(r_back < NUM_RESULTS_REQUIRED ){
        std::cout << "waiting for " <<  NUM_RESULTS_REQUIRED - r_back  << " / " << NUM_RESULTS_REQUIRED << " results...\n" << std::endl;
        select_fds = select_fds_main;
        
        int fd_back = select(max_fd+1, &select_fds, NULL, NULL, NULL);
        if (fd_back < 0){
          std::cout << "select returned on error" << std::endl;
          break;
        }
        if (fd_back>0){
          for(int fd_id =0; fd_id < NUMGPU; ++fd_id){
            if( !FD_ISSET(nodes[fd_id], &select_fds) )
              continue;
            data_block_t result;
            int val_read = read(nodes[fd_id], &result, sizeof(result));
            if(val_read >0 && result.cmd==NODE_OUTPUT_AVAILABLE){
              std::cout << "SUCCESS, GPU computation finished on worker " << result.identifier <<std::endl;
              key_t key = ftok("res",1234);            
              int shmid = shmget(key, X*Z*sizeof(float), 0660|IPC_CREAT); 
              float *data = (float*) shmat(shmid,(void*)0,0); 
              r_back++;
              shmctl(shmid,IPC_RMID,NULL); 
            }
          }
        }
    }

    // at this point, we have collected enough results, and we wish to first tell the other GPUs to stop
    // and possibly start a new set of computation before we start collecting all the results.
    std::cout << "Collected enough results to compute the results, sending interrupt to unfinised kernels" <<std::endl; 
   /* 
    
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

        }
   }
*/
}

int main(){
    int masterfd=0;
    sockaddr_in* sock = create_master_server(masterfd);
    std::cout << "waiting for " << NUMGPU << " devices to connect\n";
    // Note that the master will not start issueing command until it gets enough node connections
    for(int i =0; i < NUMGPU; ++i)
      accept_node(sock, masterfd);

    send_one_round_of_inputs();
}
