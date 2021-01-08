// Created by Ling Liang.

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#define MAX_FLOAT  (std::numeric_limits<float>::max())
extern  int TENSOR_NUM;
extern  int opt_type; // 0 find MS,  1 find MC
extern  int thread_num;

using namespace std;

typedef struct SubNetwork{
    float S;              // size of each sub-network (S)
    float *R;             // row vector of each sub-network (R)
    float CC = MAX_FLOAT; // lowest maximum contraction cost of each sub-network (CC)
    int sq[2];            // the selected split case (sq)
    float *O;             // outer product vector of each sub-network (O)
    bool P;               // 1: connective sub-network 0: sub-network composed through outer prodcut
}SubNetwork;


typedef struct Network {
    float **adj;          // adj matrix of tensor network
    float **adj_O; 
    SubNetwork *TI;       // array store all subnetwork's information
    int *adr_v;           // address of each tensor
    int step;             // used for back tracking
    int **combine;        // fast search for C_i^j use for multithread
}Network;


typedef struct TaskInf{
    int v;                // number of tensors in the set
    int id;               // the start id of each task
    int instance;         // number of instances in each task    
    Network *net_ptr;   
}TaskInf;

#endif //TENSOR_H
