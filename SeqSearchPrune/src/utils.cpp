// Created by Ling Liang.

#include <iostream>
#include <random>
#include "utils.h"

void show_matrix(float **M, int row, int col){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++) printf("%.1f ", M[i][j]);
        printf("\n");
    }
    printf("\n");
}


void adj_init(Network &net){

    net.adj = (float**) malloc(TENSOR_NUM * sizeof(float *));
    for(int i = 0; i < TENSOR_NUM; i++){
        net.adj[i] = (float*) malloc((TENSOR_NUM + 1) * sizeof(float));
    }

    // fill sharing orders and free orders in adj
    for(int i = 0; i < TENSOR_NUM; i++){
        for(int j = 0; j < TENSOR_NUM; j++){
            if(i == j) net.adj[i][j] = 0;
            else       net.adj[i][j] = 0.5;
        }
        net.adj[i][TENSOR_NUM] = 0.1;
    }

    cout << "\nadjacent matrix:\n";
    show_matrix(net.adj, TENSOR_NUM, TENSOR_NUM + 1);
}


void adj_init(Network &net, int basic_structure, float sparsity){

    net.adj = (float**) malloc(TENSOR_NUM * sizeof(float *));
    for(int i = 0; i < TENSOR_NUM; i++){
        net.adj[i] = (float*) malloc((TENSOR_NUM + 1) * sizeof(float));
        for(int j = 0; j < TENSOR_NUM; j++) net.adj[i][j] = 0;
        net.adj[i][TENSOR_NUM] = 0.1;
    }

    // shuffle tensor ID
    vector<int> TI_id(TENSOR_NUM);
    for(int i = 0; i < TENSOR_NUM; i++) TI_id[i] = i;
    shuffle(TI_id.begin(), TI_id.end(), default_random_engine(79));
    // cout << "\n";
    // for(int i = 0; i < TENSOR_NUM; i++) cout << TI_id[i] << "\t";
    // cout << "\n";

    // chain structure
    if(basic_structure == 0){
        int curr_id, next_id;
        for(int i = 0; i < TENSOR_NUM - 1; i++){
            curr_id = TI_id[i]; next_id = TI_id[i + 1];
            net.adj[curr_id][next_id] = 0.5;
            net.adj[next_id][curr_id] = 0.5;
        }
    }

    // binary tree structure
    if(basic_structure == 1){
        int curr_id, parent_id;
        for(int i = 1; i < TENSOR_NUM; i++){
            curr_id = TI_id[i]; parent_id = TI_id[(i - 1) / 2];
            net.adj[curr_id][parent_id] = 0.5;
            net.adj[parent_id][curr_id] = 0.5;
        }
    }

    // center structure
    if(basic_structure == 2){
        int center_id = TI_id[0];
        for(int i = 0; i < TENSOR_NUM; i++){
            if(i != center_id){
                net.adj[center_id][i] = 0.5;
                net.adj[i][center_id] = 0.5;
            }
        }
    }

    // mesh structure
    if(basic_structure == 3){
        int row_num, col_num;
        if(TENSOR_NUM == 16){
            row_num = 4; col_num = 4;
        }
        else if(TENSOR_NUM == 18){
            row_num = 3; col_num = 6;
        }
        else if(TENSOR_NUM == 20){
            row_num = 4; col_num = 5;
        }
        else if(TENSOR_NUM == 24){
            row_num = 4; col_num = 6;
        }
        else if(TENSOR_NUM == 25){
            row_num = 5; col_num = 5;
        }
        else{
            exit(0);
        }

        int curr_id, next_right_id, next_below_id;
        for(int i = 0; i < row_num; i++){
            for(int j = 0; j < col_num; j++){
                curr_id       = TI_id[i * col_num + j]; 
                next_right_id = TI_id[i * col_num + (j + 1)];
                next_below_id = TI_id[(i + 1) * col_num + j];

                if(j < col_num - 1){
                    net.adj[curr_id][next_right_id] = 0.5;
                    net.adj[next_right_id][curr_id] = 0.5;
                }

                if(i < row_num - 1){
                    net.adj[curr_id][next_below_id] = 0.5;
                    net.adj[next_below_id][curr_id] = 0.5;
                }

            }
        }

    }

    // fill extra sharing orders
    int total_edg = 0;

    vector<int> triangle_idx(TENSOR_NUM, 0);
    for(int i = 0; i < TENSOR_NUM - 1; i++){
        total_edg += (i + 1);
        triangle_idx[i + 1] = total_edg;
    }

    int pick_edg = int(total_edg * sparsity);
    if(pick_edg > (total_edg - TENSOR_NUM + 1)) pick_edg = TENSOR_NUM - 1;
    cout << "extra edges: " << pick_edg << "\n";

    vector<int> edg_id(total_edg);
    for(int i = 0; i < total_edg; i++) edg_id[i] = i;
    shuffle(edg_id.begin(), edg_id.end(), default_random_engine(19));

    int axis1, axis2;
    int offset = 0;
    int count = 0;


    while(count < pick_edg){
        for(int i = 0; i < TENSOR_NUM - 1; i++){
            if(edg_id[offset] >= triangle_idx[i] && edg_id[offset] < triangle_idx[i + 1]){
                axis1 = i + 1;
                axis2 = edg_id[offset] - triangle_idx[i];
                if(net.adj[axis1][axis2] == 0){
                    net.adj[axis1][axis2] = 0.5;
                    net.adj[axis2][axis1] = 0.5;
                    count ++;
                }
                break;
            }
        }
        offset ++;
    };

    cout << "\nadjacent matrix:\n";
    show_matrix(net.adj, TENSOR_NUM, TENSOR_NUM + 1);
}

void para_init(Network &net) {

    // init all sub-networks in tensor network
    int TI_len = 1;
    for(int i = 0; i < TENSOR_NUM - 1; i++) TI_len = (TI_len << 1) + 1;
    TI_len ++;
    net.TI = (SubNetwork*) malloc(TI_len * sizeof(SubNetwork));

    // init sub-networks in Set_1
    net.adr_v = (int*) malloc(TENSOR_NUM * sizeof(int));
    for(int i = 0; i < TENSOR_NUM; i++){
        net.adr_v[i] = 1<<i; // find the real adr of each tensor

        // initial sub-network size (S) & sub-network row vector (R) in Set_1
        net.TI[net.adr_v[i]].S = net.adj[i][TENSOR_NUM];
        net.TI[net.adr_v[i]].R = (float*) malloc(TENSOR_NUM * sizeof(float));
        for(int j = 0; j < TENSOR_NUM; j++){
            net.TI[net.adr_v[i]].S += (net.adj[i][j] + net.adj[j][i]);
            net.TI[net.adr_v[i]].R[j] = net.adj[i][j];
        }

        // init contraction cost (CC) based on optimization type.
        if(opt_type == 0) net.TI[net.adr_v[i]].CC = net.TI[net.adr_v[i]].S;
        else net.TI[net.adr_v[i]].CC = 0;

        // give a flag to sq[0] used in back tracking the contraction sequence
        net.TI[net.adr_v[i]].sq[0] = -1;
    }
}


void FPrune(Network &net, int v, int adr_TI) {
    // collect J do not share orders with TI
    int outer_adr = 0;
    int count = 0;
    int select_element[TENSOR_NUM];

    for(int i = 0; i < TENSOR_NUM; i++){
        if(net.TI[adr_TI].O[i] == 0){
            select_element[count] = net.adr_v[i];
            count += 1;
        }
    }

    // find all sub-network do not connect to TI
    for(int i = v; i < count + 1; i++){
        vector<bool> outer_TI(count, false);
        fill(outer_TI.end() - i,  outer_TI.end(), true);
        do{
            outer_adr = 0;
            for(int j = 0; j < count; j++){
                if(outer_TI[j]) outer_adr += select_element[j];
            }
            net.TI[outer_adr + adr_TI].P = false;
        }while(next_permutation(outer_TI.begin(), outer_TI.end()));
    }
}


chrono::duration<double> OP_init(Network &net){
    // identify outer product based on set_1
    int TI_len = 1;
    for(int i = 0; i < TENSOR_NUM - 1; i++) TI_len = (TI_len << 1) + 1;
    TI_len ++;

    for(int i = 0; i < TI_len; i++) net.TI[i].P = true;        // init the P in TI
    net.adj_O = (float**) malloc(TENSOR_NUM * sizeof(float*)); // init adj_O
    for(int i = 0; i < TENSOR_NUM; i++){
        net.adj_O[i] = (float*) malloc(TENSOR_NUM * sizeof(float));
        for(int j = 0; j < TENSOR_NUM; j++) net.adj_O[i][j] = net.adj[i][j];
        net.adj_O[i][i] = 1;
    }

    
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < TENSOR_NUM; i++){
        net.TI[net.adr_v[i]].O = (float*) malloc(TENSOR_NUM * sizeof(float));
        for(int j = 0; j < TENSOR_NUM; j++) net.TI[net.adr_v[i]].O[j] = net.adj_O[i][j] + net.adj_O[j][i];
        FPrune(net, 1, net.adr_v[i]);
    }
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> initial_prune_time = chrono::duration_cast<std::chrono::microseconds>(stop - start);

    return initial_prune_time;
}


void multithread_init(Network &net){
    // init combination matrix
    net.combine = (int**) malloc(TENSOR_NUM * sizeof(int*));
    int tmp;
    for(int i = 1; i < (TENSOR_NUM + 1); i++){
        net.combine[i - 1] = (int*) malloc(TENSOR_NUM * sizeof(int));
        for(int j = 0; j < TENSOR_NUM; j++) net.combine[i - 1][j] = 0;
        for(int j = 1; j < (i + 1); j++){
            tmp = i;
            for(int z = 1; z < j; z++){
                tmp *= (i - z);
                tmp /= z;
            } tmp /= j;
            net.combine[i - 1][j - 1] = tmp;
        }
    }
}


void back_track_sq(Network &net, int **sq, float *cc, int adr){
    if(net.TI[adr].sq[0] != -1){
        sq[0][net.step] = net.TI[adr].sq[0];
        sq[1][net.step] = net.TI[adr].sq[1];
        cc[net.step] = net.TI[adr].CC;
        net.step --;
        back_track_sq(net, sq, cc, net.TI[adr].sq[0]);
        back_track_sq(net, sq, cc, net.TI[adr].sq[1]);
    }
}


void get_seq(Network &net) {

    // back track the contraction sequence
    int **sq;
    float *cc;
    sq = (int**) malloc(2 * sizeof(int *));
    cc = (float*) malloc((TENSOR_NUM - 1) * sizeof(float));

    for(int i = 0; i < 2; i++){
        sq[i] = (int*) malloc((TENSOR_NUM - 1) * sizeof(int));
    }
    net.step = TENSOR_NUM - 2;

    int TI_len = 1;
    for(int i = 0; i < TENSOR_NUM - 1; i++) TI_len = (TI_len << 1) + 1;

    back_track_sq(net, sq, cc, TI_len);

    cout << "contraction sequence:\n";
    for(int i = 0; i < TENSOR_NUM - 1; i++) {
        //cout << "step" << i+1 << ":\t" << sq[0][i] << "\t\t" << sq[1][i]<< "\n";
        cout << "step" << i+1 << ":\t" << sq[0][i] << "\t\t" << sq[1][i]<< "\t\t" << cc[i] << "\n";
    }
}
