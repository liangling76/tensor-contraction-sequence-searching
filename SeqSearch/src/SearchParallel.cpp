// Created by Ling Liang.

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "utils.h"


void find_start(int n, int r, int id, vector<bool> &start, Network *net_ptr){
    int tmp_n = n - 1;
    for(int i = 0; i < n - 1; i++){
        tmp_n --;
        if(r > 0){
            if (id <= net_ptr->combine[tmp_n][r - 1]){
                start[i] = false;
            }
            else{
                start[i] = true;
                id -= net_ptr->combine[tmp_n][r - 1];
                r--;
            }
        }
    }
    if(r > 0) start[n - 1] = true;
}


void *PMS(void *arg) {
    auto *task_inf = static_cast<TaskInf*>(arg);
    int v = task_inf->v;
    int id = task_inf->id;
    int instance = task_inf->instance;
    Network *net_ptr = task_inf->net_ptr;

    // init the bit map of first TI in Set_v
    vector<bool> bitmap_TI(TENSOR_NUM, false);
    find_start(TENSOR_NUM, v, id, bitmap_TI, net_ptr);

    // other parameter initialization
    int adr_TI, adr_TP1, adr_TP2;                     // physical address of sub-network TI and two split-networks TP1 TP2
    auto *adr_TI_a = (int*) malloc(v * sizeof(int));  // physical address of tensors in TI
    auto *idx_TI_a = (int*) malloc(v * sizeof(int));  // tensor ID in TI
    vector<bool> bitmap_TP1(v, false);                // vector used to split the sub-network
    float SO;                                         // sharing orders between two split-networks
    float Exp;                                        // Exp: contraction expense between TP1 and TP2
    float tmp_CC, opt_CC;                             // tmp_CC: store temporal contraction cost, opt_CC temporal optimal contraction cost
    
    int opt_sq[2];
    int offset0;
    int instance_count = 0;

    // go through all sub-networks TI in Set_v
    do{
        adr_TI = 0;
        offset0 = 0;
        for(int i = 0; i < TENSOR_NUM; i++){
            if(bitmap_TI[i]){
                adr_TI |= (1<<i);
                adr_TI_a[offset0] = net_ptr->adr_v[i];
                idx_TI_a[offset0] = i;
                offset0 ++;
            }
        }

        // select first split case to calculate data size (S) and row vector (R) of TI
        adr_TP1 = adr_TI_a[0];
        adr_TP2 = adr_TI - adr_TP1;
        
        // calculate SO in first spilt case
        SO = net_ptr->TI[adr_TP2].R[idx_TI_a[0]];
        for(int i = 1; i < v; i++) SO += net_ptr->TI[adr_TP1].R[idx_TI_a[i]];

        // calulate tensor size (S) and row vector (R) of TI
        net_ptr->TI[adr_TI].S = net_ptr->TI[adr_TP1].S + net_ptr->TI[adr_TP2].S - 2 * SO;
        net_ptr->TI[adr_TI].R = (float*) malloc(TENSOR_NUM * sizeof(float));
        for(int i = 0; i < TENSOR_NUM; i++) net_ptr->TI[adr_TI].R[i] = net_ptr->TI[adr_TP1].R[i] + net_ptr->TI[adr_TP2].R[i];
        
        // calculate Exp
        Exp = net_ptr->TI[adr_TI].S;

        // go through all split cases of TI
        opt_CC = MAX_FLOAT;
        for(int TP1_len = 0; TP1_len < v/2; TP1_len++){
            // Equation 11 in section II.B
            fill(bitmap_TP1.begin(), bitmap_TP1.end(), false);
            if(v % 2 == 0 && TP1_len == (v / 2 - 1)){
                fill(bitmap_TP1.end() - TP1_len, bitmap_TP1.end(), true);
                bitmap_TP1[0] = true;
            }
            else fill(bitmap_TP1.end() - (TP1_len + 1), bitmap_TP1.end(), true);

            // go through all spilt cases under TP1 contains TP1_len + 1 tensors
            do{
                adr_TP1 = 0; adr_TP2 = 0;
                for(int i = 0; i < v; i++){
                    if(bitmap_TP1[i]){
                        adr_TP1 += adr_TI_a[i];
                    }
                    else{
                        adr_TP2 += adr_TI_a[i];
                    }
                }

                // calculate tmp_CC and
                tmp_CC = Exp;
                if(tmp_CC < net_ptr->TI[adr_TP1].CC) tmp_CC = net_ptr->TI[adr_TP1].CC;
                if(tmp_CC < net_ptr->TI[adr_TP2].CC) tmp_CC = net_ptr->TI[adr_TP2].CC;
                if(tmp_CC < opt_CC){
                    opt_CC = tmp_CC;
                    opt_sq[0] = adr_TP1; opt_sq[1] = adr_TP2;
                }

                if(tmp_CC == Exp) break;

            }while(next_permutation(bitmap_TP1.begin(), bitmap_TP1.end()));
        }
        net_ptr->TI[adr_TI].CC  = opt_CC;
        net_ptr->TI[adr_TI].sq[0] = opt_sq[0]; net_ptr->TI[adr_TI].sq[1] = opt_sq[1];
        instance_count ++;
    }while(instance_count < instance && next_permutation(bitmap_TI.begin(), bitmap_TI.end()));
    return nullptr;
}


void *PTC(void *arg){
    auto *task_inf = static_cast<TaskInf*>(arg);
    int v = task_inf->v;
    int id = task_inf->id;
    int instance = task_inf->instance;
    Network *net_ptr = task_inf->net_ptr;

    // init the bit map of first TI in Set_v
    vector<bool> bitmap_TI(TENSOR_NUM, false);
    find_start(TENSOR_NUM, v, id, bitmap_TI, net_ptr);

    // other parameter initialization
    int adr_TI, adr_TP1, adr_TP2;                     // physical address of sub-network TI and two split-networks TP1 TP2
    auto *adr_TI_a = (int*) malloc(v * sizeof(int));  // physical address of tensors in TI
    auto *idx_TI_a = (int*) malloc(v * sizeof(int));  // tensor ID in TI
    auto *idx_TP_a = (int*) malloc(v * sizeof(int));  // tensor ID for two split-networks
    vector<bool> bitmap_TP1(v, false);                // vector used to split the sub-network
    float SO;                                         // sharing orders between two split-networks
    float tmp_S, Exp;                                 // tmp_S = S_{T_I},  Exp: contraction expense between TP1 and TP2
    float tmp_CC, opt_CC;                             // tmp_CC: store temporal contraction cost, opt_CC temporal optimal contraction cost
    
    int opt_sq[2];
    int offset0, offset1, offset2;
    int instance_count = 0;

    // go through all sub-networks TI in Set_v
    do{
        adr_TI = 0;
        offset0 = 0;
        for(int i = 0; i < TENSOR_NUM; i++){
            if(bitmap_TI[i]){
                adr_TI |= (1<<i);
                adr_TI_a[offset0] = net_ptr->adr_v[i];
                idx_TI_a[offset0] = i;
                offset0 ++;
            }
        }


        // select one split case to calculate data size (S) and row vector (R) of TI
        adr_TP1 = adr_TI_a[0];
        adr_TP2 = adr_TI - adr_TP1;

        // calculate SO in first split case 
        SO = net_ptr->TI[adr_TP2].R[idx_TI_a[0]];
        for(int i = 1; i < v; i++) SO += net_ptr->TI[adr_TP1].R[idx_TI_a[i]];

        // calulate tensor size (S) and row vector (R) of TI
        net_ptr->TI[adr_TI].S = net_ptr->TI[adr_TP1].S + net_ptr->TI[adr_TP2].S - 2 * SO;
        net_ptr->TI[adr_TI].R = (float*) malloc(TENSOR_NUM * sizeof(float));
        for(int i = 0; i < TENSOR_NUM; i++) net_ptr->TI[adr_TI].R[i] = net_ptr->TI[adr_TP1].R[i] + net_ptr->TI[adr_TP2].R[i];
        
        // calculate Exp
        tmp_S = net_ptr->TI[adr_TI].S;


        // go through all split cases of TI
        opt_CC = MAX_FLOAT;
        for(int TP1_len = 0; TP1_len < v/2; TP1_len++){
            // Equation 11 in section II.B
            fill(bitmap_TP1.begin(), bitmap_TP1.end(), false);
            if(v % 2 == 0 && TP1_len == (v / 2 - 1)){
                fill(bitmap_TP1.end() - TP1_len, bitmap_TP1.end(), true);
                bitmap_TP1[0] = true;
            }
            else fill(bitmap_TP1.end() - (TP1_len + 1), bitmap_TP1.end(), true);

            // go through all spilt cases under TP1 contains TP1_len + 1 tensors
            do{
                adr_TP1 = 0; adr_TP2 = 0; offset1 = 0; offset2 = v - 1;
                for(int i = 0; i < v; i++){
                    if(bitmap_TP1[i]){
                        adr_TP1 += adr_TI_a[i];
                        idx_TP_a[offset1] = idx_TI_a[i];
                        offset1 ++;
                    }
                    else{
                        adr_TP2 += adr_TI_a[i];
                        idx_TP_a[offset2] = idx_TI_a[i];
                        offset2 --;
                    }
                }

                SO = 0;
                for(int i = 0; i < offset1; i++) SO += net_ptr->TI[adr_TP2].R[idx_TP_a[i]];
                for(int i = v - 1; i > offset2; i--) SO += net_ptr->TI[adr_TP1].R[idx_TP_a[i]];
                Exp = tmp_S + SO;

                // calculate tmp_CC 
                tmp_CC = Exp;
                if(tmp_CC < net_ptr->TI[adr_TP1].CC) tmp_CC = net_ptr->TI[adr_TP1].CC;
                if(tmp_CC < net_ptr->TI[adr_TP2].CC) tmp_CC = net_ptr->TI[adr_TP2].CC;
                if(tmp_CC < opt_CC){
                    opt_CC = tmp_CC;
                    opt_sq[0] = adr_TP1; opt_sq[1] = adr_TP2;
                }
            }while(next_permutation(bitmap_TP1.begin(), bitmap_TP1.end()));
        }
        net_ptr->TI[adr_TI].CC  = opt_CC;
        net_ptr->TI[adr_TI].sq[0] = opt_sq[0]; net_ptr->TI[adr_TI].sq[1] = opt_sq[1];
        instance_count++;
    }while(instance_count < instance && next_permutation(bitmap_TI.begin(), bitmap_TI.end()));
    return nullptr;
}


void scheduler(Network &net){
    
    // parameters for parallelism arrangement
    Network *net_ptr = &net;
    std::vector<TaskInf> task_infs;

    int comb, tmp_thread_num, remainder, instance;
    int *thread_instance, *thread_start_id;
    thread_instance = (int*) malloc(thread_num * sizeof(int));
    thread_start_id = (int*) malloc(thread_num * sizeof(int));
    thread_start_id[0] = 1;

    // go through all sets from 2 to V (TENSOR_NUM)
    for(int v = 2; v < TENSOR_NUM + 1; v++){
        cout << "begin Set: " << v << "\n";

        // calculate how many combinations
        tmp_thread_num = thread_num;
        comb = net.combine[TENSOR_NUM - 1][v - 1];
        if(comb < thread_num) tmp_thread_num = comb;

        instance = int(ceil( float(comb) / tmp_thread_num));
        remainder = tmp_thread_num * instance - comb;

        for(int i = 0; i < tmp_thread_num - remainder; i++) thread_instance[i] = instance;
        for(int i = 0; i < remainder; i++) thread_instance[i + tmp_thread_num - remainder] = instance - 1;
        for(int i = 1; i < tmp_thread_num; i++) thread_start_id[i] = thread_start_id[i - 1] + thread_instance[i - 1];


        task_infs.resize(tmp_thread_num);
        for(size_t i = 0; i < thread_num; i++){
            task_infs[i].v = v;
            task_infs[i].id = thread_start_id[i];
            task_infs[i].instance = thread_instance[i];
            task_infs[i].net_ptr = net_ptr;
        }

        auto p_arr = (pthread_t* ) malloc(tmp_thread_num * sizeof(pthread_t));
        for(int i = 0; i < tmp_thread_num; i++){
            if(opt_type == 0) pthread_create(&p_arr[i], nullptr, PMS, &task_infs[i]);
            else pthread_create(&p_arr[i], nullptr, PTC, &task_infs[i]);
        }
        for(int i = 0; i < tmp_thread_num; i++) pthread_join(p_arr[i], nullptr);
    }
}

void run_parallel(Network &net) {
    para_init(net);
    multithread_init(net);
    cout << "begin searching parallel...\n";

    auto start = chrono::high_resolution_clock::now();
    scheduler(net);
    auto stop = chrono::high_resolution_clock::now();

    int TI_len = 1;
    for(int i = 0; i < TENSOR_NUM - 1; i++) TI_len = (TI_len << 1) + 1;

    cout << "\nsearching result:";
    if(opt_type==0) cout << "contraction cost (storage):\t" << net.TI[TI_len].CC << "\n";
    else cout << "\ncontraction cost (computation):\t" << net.TI[TI_len].CC << "\n";
    cout << "execution time:\t" << chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "us\n\n";

}
