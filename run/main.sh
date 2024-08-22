#!/bin/bash
#path="../../../../work/log1/darius.weber/Quadratic_Programming_Datasets"
path="Quadratic_Programming_Datasets"

#LARGE SVM:
#Gen_large_svm_400ins_32batch_l2 (sweep 18)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gen_large_svm_400ins_32batch_l2 --batchsize=32 --conv=genconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=178 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=0.4726154974906576 --loss_weight_obj=9.456393323035776 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=3 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gen_large_svm_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp6/ --hidden 178  --conv genconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 3 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

#Gin_large_svm_400ins_32batch_l2 (sweep 49)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gin_large_svm_400ins_32batch_l2 --batchsize=32 --conv=ginconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=262 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=2.975650732894602 --loss_weight_obj=8.185407711754467 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gin_large_svm_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp7/ --hidden 262  --conv ginconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

#Gcn_large_svm_400ins_32batch_l2 (sweep 33)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gcn_large_svm_400ins_32batch_l2  --batchsize=32 --conv=gcnconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=109 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=6.599643920659866 --loss_weight_obj=9.980468919867668 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gcn_large_svm_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp8/ --hidden 109  --conv gcnconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets



#MARKOWITZ PORTFOLIO:
#Gen_large_portfolio_400ins_32batch_l2 (sweep 46)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gen_large_portfolio_400ins_32batch_l2 --batchsize=32 --conv=genconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=144 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=6.374064663942023 --loss_weight_obj=9.91718499504328 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gen_large_portfolio_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp9/ --hidden 144  --conv genconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

#Gin_large_portfolio_400ins_32batch_l2 (sweep 13)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gin_large_portfolio_400ins_32batch_l2 --batchsize=32 --conv=ginconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=202 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=3.17245010510955 --loss_weight_obj=0.229641202140837 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gin_large_portfolio_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp10/ --hidden 202  --conv ginconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

#Gcn_large_portfolio_400ins_32batch_l2 (sweep 36)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gcn_large_portfolio_400ins_32batch_l2 --batchsize=32 --conv=gcnconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=264 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=2.725279545442537 --loss_weight_obj=1.283431213447762 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gcn_large_portfolio_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp11/ --hidden 264  --conv gcnconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets



#LARGE REGRESSION:
#Gen_large_regression_400ins_32batch_l2 (sweep 40)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gen_large_regression_400ins_32batch_l2 --batchsize=32 --conv=genconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=150 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=5.945342296967018 --loss_weight_obj=2.887496329479448 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gen_large_regression_400ins_32batch_l2  --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp3/ --hidden 150  --conv genconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

#Gin_large_regression_400ins_32batch_l2 (sweep 23)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gin_large_regression_400ins_32batch_l2 --batchsize=32 --conv=ginconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=145 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=3.536993245681094 --loss_weight_obj=3.532940135838206 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gin_large_regression_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp4/ --hidden 145  --conv ginconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

#Gcn_large_regression_400ins_32batch_l2 (sweep 33)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gcn_large_regression_400ins_32batch_l2 --batchsize=32 --conv=gcnconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=109 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=6.599643920659866 --loss_weight_obj=9.980468919867668 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gcn_large_regression_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp5/ --hidden 109  --conv gcnconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets



#FITTING CONVEX:
#Gen_large_fittingconvex_400ins_32batch_l2 (sweep 31)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gen_large_fittingconvex_400ins_32batch_l2 --batchsize=32 --conv=genconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=106 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=1.6697621098201474 --loss_weight_obj=7.107878509000387 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=3 --runs=3 --weight_decay=0
#python evaluate.py  --wandbname=Gen_large_fittingconvex_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp0/ --hidden 106  --conv genconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 3 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

#Gin_large_fittingconvex_400ins_32batch_l2 (sweep 28)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gin_large_fittingconvex_400ins_32batch_l2 --batchsize=32 --conv=ginconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=236 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=2.80634291757392 --loss_weight_obj=1.7086627382229107 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=3 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gin_large_fittingconvex_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp1/ --hidden 236  --conv ginconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 3 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

#Gcn_large_fittingconvex_400ins_32batch_l2 (sweep 24)
#python main.py --use_wandb=True --wandbproject=EVAL_GNNs --epoch=1000 --wandbname=Gcn_large_fittingconvex_400ins_32batch_l2 --batchsize=32 --conv=gcnconv --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets --hidden=172 --ipm_alpha=0.8 --ipm_steps=8 --loss_weight_cons=9.40396835558935 --loss_weight_obj=1.2896698795535009 --loss_weight_x=1 --losstype=l2 --lr=0.001 --micro_batch=1 --num_mlp_layers=2 --num_pred_layers=4 --runs=3 --weight_decay=0
#python evaluate.py --wandbname=Gcn_large_fittingconvex_400ins_32batch_l2 --modelpath ../../../../work/log1/darius.weber/FINALlogs/exp2/ --hidden 172  --conv gcnconv --use_wandb=True --wandbproject=FINAL_TIME_GNNs --ipm_steps 8 --num_pred_layers 4 --num_mlp_layers 2 --conv_sequence cov --datapath=../../../../work/log1/darius.weber/Quadratic_Programming_Datasets

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