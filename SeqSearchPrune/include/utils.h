// Created by Ling Liang.

#ifndef UTILS_H
#define UTILS_H
#include "Tensor.h"

void show_matrix(float **M, int row, int col);      // show an adj matrix

void adj_init(Network &net);                        // init a fully connected tensor network
void adj_init(Network &net, int basic_structure, float sparsity);

void para_init(Network &net);                       // init the parameters of the tensor network
chrono::duration<double> OP_init(Network &net);     // init the parameters used for outer product
void multithread_init(Network &net);                // init the parameters for multithread

void FPrune(Network &net, int v, int adr_TI);       // used to search TI which composed through outer product
void run(Network &net);                             // search optimal contraction sequence
void run_parallel(Network &net);                    // search optimal contraction sequence through multithread

void get_seq(Network &net);                         // get the contraction sequence


#endif //UTILS_H