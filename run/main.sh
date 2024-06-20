#!/bin/bash

python main.py --datapath Quadratic_Programming_Datasets --ipm_alpha 0.15 --weight_decay 1.2e-6 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --conv gcnconv


# Function to handle termination based on input
terminate_on_t() {
    while :; do
        read -n 1 -s input
        if [[ $input == 't' ]]; then
            echo "Terminating script."
            break
        echo "Press 't' to terminate."
        fi
        echo "Invalid input. Try again."
    done
}
terminate_on_t