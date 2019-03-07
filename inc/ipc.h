#ifndef NODE_H
#define NODE_H

#define PORT 8080
#ifndef NUMGPU
#define NUMGPU 1
#endif

enum IPCCommand {
    NODE_CONNECT,
    MASTER_ACK,
    MASTER_INPUT_AVAILABLE,
    NODE_OUTPUT_AVAILABLE,
    MASTER_NODE_SHUTDOWN,
    NODE_KERNEL_ERR
};

#endif
