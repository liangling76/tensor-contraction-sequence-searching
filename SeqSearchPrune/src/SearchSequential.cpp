// Created by Ling Liang.

#include <iostream>
#include <vector>
#include <chrono>
#include "utils.h"


void MSseq(Network &net){
    // go through all sets from 2 to V (TENSOR_NUM)
    for(int v = 2; v < TENSOR_NUM + 1; v++){
        cout << "begin Set: " << v << "\n";

        // init the bit map of first TI in Set_v
        vector<bool> bitmap_TI(TENSOR_NUM, false);
        fill(bitmap_TI.end() - v, bitmap_TI.end(), true);

        // other parameter initialization
        int adr_TI, adr_TP1, adr_TP2;                     // decimal address of sub-network TI and two split-networks TP1 TP2
        auto *adr_TI_a = (int*) malloc(v * sizeof(int));  // decimal address of tensors in TI
        auto *idx_TI_a = (int*) malloc(v * sizeof(int));  // tensor ID in TI
        auto *idx_TP_a = (int*) malloc(v * sizeof(int));  // tensor ID for two split-networks
        vector<bool> bitmap_TP1(v, false);                // vector used to split the sub-network
        float SO;                                         // sharing orders between two split-networks
        float Exp;                                        // Exp: contraction expense between TP1 and TP2
        float tmp_CC, opt_CC;                             // tmp_CC: store temporal contraction cost, opt_CC temporal optimal contraction cost

        int opt_sq[2];
        int suc_sp, offset0, offset1, offset2;

        // go through all sub-networks TI in Set_v
        do{
            adr_TI = 0;
            offset0 = 0;
            for(int i = 0; i < TENSOR_NUM; i++){
                if(bitmap_TI[i]){
                    adr_TI |= (1<<i);
                    adr_TI_a[offset0] = net.adr_v[i];
                    idx_TI_a[offset0] = i;
                    offset0 ++;
                }
            }

            if(net.TI[adr_TI].P){
                // go through all split cases of TI
                suc_sp = false;
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

                        if(net.TI[adr_TP1].P && net.TI[adr_TP2].P){
                            if(!suc_sp){
                                // update information of TI
                                offset1 = 0; offset2 = v - 1;
                                for(int i = 0; i < v; i++){
                                    if(bitmap_TP1[i]){
                                        idx_TP_a[offset1] = idx_TI_a[i];
                                        offset1 ++;
                                    }
                                    else{
                                        idx_TP_a[offset2] = idx_TI_a[i];
                                        offset2 --;
                                    }
                                }

                                // calculate the SO in current case
                                SO = 0;
                                for(int i = 0; i < offset1; i++) SO += net.TI[adr_TP2].R[idx_TP_a[i]];
                                for(int i = v - 1; i > offset2; i--) SO += net.TI[adr_TP1].R[idx_TP_a[i]];

                                // calulate tensor size (S), row vector (R) of TI, and Exp 
                                net.TI[adr_TI].S = net.TI[adr_TP1].S + net.TI[adr_TP2].S - 2 * SO;
                                net.TI[adr_TI].R = (float*) malloc(TENSOR_NUM * sizeof(float));
                                net.TI[adr_TI].O = (float*) malloc(TENSOR_NUM * sizeof(float));
                                for(int i = 0; i < TENSOR_NUM; i++) net.TI[adr_TI].R[i] = net.TI[adr_TP1].R[i] + net.TI[adr_TP2].R[i];
                                for(int i = 0; i < TENSOR_NUM; i++) net.TI[adr_TI].O[i] = net.TI[adr_TP1].O[i] + net.TI[adr_TP2].O[i];
                                Exp = net.TI[adr_TI].S;

                                suc_sp = true;
                            }

                            // calculate tmp_CC
                            tmp_CC = Exp;
                            if(tmp_CC < net.TI[adr_TP1].CC) tmp_CC = net.TI[adr_TP1].CC;
                            if(tmp_CC < net.TI[adr_TP2].CC) tmp_CC = net.TI[adr_TP2].CC;
                            if(tmp_CC < opt_CC){
                                opt_CC = tmp_CC;
                                opt_sq[0] = adr_TP1; opt_sq[1] = adr_TP2;
                            }

                            if (tmp_CC == Exp) break;
                        }
                    }while(next_permutation(bitmap_TP1.begin(), bitmap_TP1.end()));
                }
                net.TI[adr_TI].CC  = opt_CC;
                net.TI[adr_TI].sq[0] = opt_sq[0]; net.TI[adr_TI].sq[1] = opt_sq[1];
                FPrune(net, v, adr_TI);
            }
        }while(next_permutation(bitmap_TI.begin(), bitmap_TI.end()));
    }
}


void MCseq(Network &net){
    // go through all sets from 2 to V (TENSOR_NUM)
    for(int v = 2; v < TENSOR_NUM + 1; v++){
        cout << "begin Set: " << v << "\n";

        // init the bit map of first TI in Set_v
        vector<bool> bitmap_TI(TENSOR_NUM, false);
        fill(bitmap_TI.end() - v, bitmap_TI.end(), true);

        // other parameter initialization
        int adr_TI, adr_TP1, adr_TP2;                     // decimal address of sub-network TI and two split-networks TP1 TP2
        auto *adr_TI_a = (int*) malloc(v * sizeof(int));  // decimal address of tensors in TI
        auto *idx_TI_a = (int*) malloc(v * sizeof(int));  // tensor ID in TI
        auto *idx_TP_a = (int*) malloc(v * sizeof(int));  // tensor ID for two split-networks
        vector<bool> bitmap_TP1(v, false);                // vector used to split the sub-network
        float SO;                                         // sharing orders between two split-networks
        float tmp_S, Exp;                                 // Exp: contraction expense between TP1 and TP2
        float tmp_CC, opt_CC;                             // tmp_CC: store temporal contraction cost, opt_CC temporal optimal contraction cost

        int opt_sq[2];
        int suc_sp, offset0, offset1, offset2;

        // go through all sub-networks TI in Set_v
        do{
            adr_TI = 0;
            offset0 = 0;
            for(int i = 0; i < TENSOR_NUM; i++){
                if(bitmap_TI[i]){
                    adr_TI |= (1<<i);
                    adr_TI_a[offset0] = net.adr_v[i];
                    idx_TI_a[offset0] = i;
                    offset0 ++;
                }
            }

            if(net.TI[adr_TI].P){
                // go through all split cases of TI
                suc_sp = false;
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

                        if(net.TI[adr_TP1].P && net.TI[adr_TP2].P){
                            if(!suc_sp){
                                // update information of TI
                                // calculate the SO in current case
                                SO = 0;
                                for(int i = 0; i < offset1; i++) SO += net.TI[adr_TP2].R[idx_TP_a[i]];
                                for(int i = v - 1; i > offset2; i--) SO += net.TI[adr_TP1].R[idx_TP_a[i]];

                                // calulate tensor size (S), row vector (R) of TI, and Exp 
                                net.TI[adr_TI].S = net.TI[adr_TP1].S + net.TI[adr_TP2].S - 2 * SO;
                                net.TI[adr_TI].R = (float*) malloc(TENSOR_NUM * sizeof(float));
                                net.TI[adr_TI].O = (float*) malloc(TENSOR_NUM * sizeof(float));
                                for(int i = 0; i < TENSOR_NUM; i++) net.TI[adr_TI].R[i] = net.TI[adr_TP1].R[i] + net.TI[adr_TP2].R[i];
                                for(int i = 0; i < TENSOR_NUM; i++) net.TI[adr_TI].O[i] = net.TI[adr_TP1].O[i] + net.TI[adr_TP2].O[i];
                                tmp_S = net.TI[adr_TI].S;

                                suc_sp = true;
                            }
                            
                            SO = 0;
                            for(int i = 0; i < offset1; i++) SO += net.TI[adr_TP2].R[idx_TP_a[i]];
                            for(int i = v - 1; i > offset2; i--) SO += net.TI[adr_TP1].R[idx_TP_a[i]];
                            Exp = tmp_S + SO;

                            // calculate tmp_CC
                            tmp_CC = Exp;
                            if(tmp_CC < net.TI[adr_TP1].CC) tmp_CC = net.TI[adr_TP1].CC;
                            if(tmp_CC < net.TI[adr_TP2].CC) tmp_CC = net.TI[adr_TP2].CC;
                            if(tmp_CC < opt_CC){
                                opt_CC = tmp_CC;
                                opt_sq[0] = adr_TP1; opt_sq[1] = adr_TP2;
                            }
                        }
                    }while(next_permutation(bitmap_TP1.begin(), bitmap_TP1.end()));
                }
                net.TI[adr_TI].CC  = opt_CC;
                net.TI[adr_TI].sq[0] = opt_sq[0]; net.TI[adr_TI].sq[1] = opt_sq[1];
                FPrune(net, v, adr_TI);
            }
        }while(next_permutation(bitmap_TI.begin(), bitmap_TI.end()));
    }
}


void run(Network &net) {
    para_init(net);
    chrono::duration<double> initial_prune_time = OP_init(net);
    cout << "begin searching...\n";

    auto start = chrono::high_resolution_clock::now();
    if(opt_type==0) MSseq(net);
    else MCseq(net);
    auto stop = chrono::high_resolution_clock::now();

    int TI_len = 1;
    for(int i = 0; i < TENSOR_NUM - 1; i++) TI_len = (TI_len << 1) + 1;

    cout << "\nsearching result:\n";
    if(opt_type==0) cout << "contraction cost (storage):\t" << net.TI[TI_len].CC << "\n";
    else cout << "\ncontraction cost (computation):\t" << net.TI[TI_len].CC << "\n";
    cout << "execution time:\t" << initial_prune_time.count() + chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1e6 << " s\n\n";

}
