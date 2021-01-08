// Created by Ling Liang.

#include <iostream>
#include "utils.h"

int TENSOR_NUM = 4;   // number of tensors in the network (V)
int opt_type = 0;
int thread_num = 1;

int main(){

    printf("Enter tensor number: ");
    scanf("%d", &TENSOR_NUM);
    printf("Select optimization type (0 MS, 1 MC): ");
    scanf("%d", &opt_type);
    printf("Number of threads: ");
    scanf("%d", &thread_num);

    Network net; 
	// adj_init(net);

    int structure;
    float prune_portion;
    printf("base structure: ");
    scanf("%d", &structure);
    printf("prune_portion: ");
    scanf("%f", &prune_portion);
    adj_init(net, structure, prune_portion);

    if(thread_num == 1) run(net);
    else if(thread_num >= 1) run_parallel(net);

    // get_seq(net);

    return 0;
}

