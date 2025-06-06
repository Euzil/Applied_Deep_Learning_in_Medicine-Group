#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=logs/dataset_221_3d_evaluate.out
#SBATCH --error=logs/dataset_221_3d_evaluate.err
#SBATCH --time=0-24:00:00
#SBATCH --partition=asteroids
#SBATCH --qos=master
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=go73hay@mytum.de
#SBATCH --mail-type=ALL

# 加载环境
source ~/.bashrc
conda init bash
conda activate ADML

# 设置环境变量
export PYTHONPATH="/vol/miltank/users/wyou/Documents/nnUNet:$PYTHONPATH"
export nnUNet_raw="/u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/nnUNet_raw"
export nnUNet_preprocessed="/u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/nnUNet_preprocessed"
export nnUNet_results="/u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/nnUNet_trained_models"

# 解决多进程问题的关键设置
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 # training 4
export MKL_NUM_THREADS=1 # training 4
export TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_system
export CUDA_LAUNCH_BLOCKING=1
export NUM_THREADS=1
# 设置Python缓冲
export PYTHONUNBUFFERED=1

# 预处理
# nnUNetv2_plan_and_preprocess -d 221 --verify_dataset_integrity
# 启动训练 低分辨率
# nnUNetv2_train 221 3d_lowres 0
# 启动训练 平均满分辨率
# nnUNetv2_train 221 3d_fullres 1
# 预测
# nnUNetv2_predict -d 221 -i /u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/nnUNet_raw/Dataset221_KiTS2023/imagesTs/ -o /u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/predictions_fold2 -f 1 -c 3d_fullres --save_probabilities
# 评估
nnUNetv2_evaluate_folder \
  -djfile /u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/nnUNet_preprocessed/Dataset221_KiTS2023/dataset.json \
  -pfile /u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/nnUNet_preprocessed/Dataset221_KiTS2023/nnUNetPlans.json \
  /u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/nnUNet_raw/Dataset221_KiTS2023/labelsTs \
  /u/home/wyou/Documents/nnUNet/nnUNetFrame/dataset_221/predictions_fold2