// Created by Ling Liang.

#ifndef UTILS_H
#define UTILS_H
#include "Tensor.h"

void show_matrix(float **M, int row, int col);      // show an adj matrix

void adj_init(Network &net);                        // init a fully connected tensor network
void adj_init(Network &net, int basic_structure, float sparsity);

void para_init(Network &net);                       // init the parameters of the tensor network
void multithread_init(Network &net);                 // init the parameters for multithread

void run(Network &net);                             // search optimal contraction sequence
void run_parallel(Network &net);                    // search optimal contraction sequence through multithread

void get_seq(Network &net);                         // get the contraction sequence


#endif //UTILS_H