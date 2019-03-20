#ifndef NODE_H
#define NODE_H

#define PORT 8080
#ifndef NUMGPU
#define NUMGPU 1
#endif

#define X 3
#define Y 30
#define Z 30

enum IPCCommand {
    NODE_CONNECT,
    MASTER_ACK,
    MASTER_INPUT_AVAILABLE,
    NODE_OUTPUT_AVAILABLE,
    MASTER_NODE_SHUTDOWN,
    NODE_KERNEL_ERR
};


// this is the container for transfering data to and from
// the master and worker nodes. 
typedef struct data_block{
  IPCCommand cmd;
  int identifier;
} data_block_t;

#endif
