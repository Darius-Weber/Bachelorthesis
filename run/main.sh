#!/bin/bash
#path="../../../../work/log1/darius.weber/Quadratic_Programming_Datasets"
path="Quadratic_Programming_Datasets"
# python main.py --datapath Quadratic_Programming_Datasets --ipm_alpha 0.15 --weight_decay 1.2e-6 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --conv gcnconv
# python main.py --datapath Quadratic_Programming_Datasets --ipm_alpha 0.8 --weight_decay 3.8e-6 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.99 --loss_weight_cons 8.15 --runs 3 --conv genconv
python main.py --datapath $path --ipm_alpha 0.73 --weight_decay 5.6e-6 --batchsize 512 --hidden 180 --num_pred_layers 3 --num_mlp_layers 2 --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.4 --loss_weight_cons 7.5 --runs 3 --conv ginconv
#python main.py --datapath ../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --ipm_alpha 0.73 --weight_decay 5.6e-6 --batchsize 512 --hidden 180 --num_pred_layers 3 --num_mlp_layers 2 --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.4 --loss_weight_cons 7.5 --runs 3 --conv ginconv

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