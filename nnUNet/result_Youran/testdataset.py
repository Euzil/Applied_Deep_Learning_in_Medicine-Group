#!/usr/bin/env python3
"""
为 nnU-Net 创建自定义训练/测试集划分
将数据集前半部分作为训练集，后半部分作为测试集
"""

import os
import json
import shutil
from pathlib import Path
import numpy as np

def create_custom_split_dataset(source_dataset_path, output_path, dataset_id=221):
    """
    创建自定义划分的数据集
    
    Args:
        source_dataset_path: 原始数据集路径 (Dataset220_KiTS2023)
        output_path: 输出路径 (nnUNet_raw)
        dataset_id: 新数据集ID
    """

    
    
    source_path = Path(source_dataset_path)
    new_dataset_name = f"Dataset{dataset_id:03d}_KiTS2023_CustomSplit"
    target_path = Path(output_path) / new_dataset_name
    
    # 创建目录结构
    images_tr = target_path / "imagesTr"
    labels_tr = target_path / "labelsTr"
    images_ts = target_path / "imagesTs"
    labels_ts = target_path / "labelsTs"  # 用于存储测试集标签（评估用）
    
    for dir_path in [images_tr, labels_tr, images_ts, labels_ts]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"创建新数据集: {new_dataset_name}")
    
    # 获取所有case文件
    image_files = sorted(list((source_path / "imagesTr").glob("*.nii.gz")))
    label_files = sorted(list((source_path / "labelsTr").glob("*.nii.gz")))
    
    total_cases = len(image_files)
    split_point = 400
    
    print(f"总案例数: {total_cases}")
    print(f"训练集: 前 {split_point} 个案例 (case 0-{split_point-1})")
    print(f"测试集: 后 {total_cases - split_point} 个案例 (case {split_point}-{total_cases-1})")
    
    # 复制训练集 (前半部分)
    train_count = 0
    for i in range(split_point):
        if i < len(image_files) and i < len(label_files):
            # 复制图像
            src_image = image_files[i]
            dst_image = images_tr / f"KiTS_{train_count:03d}_0000.nii.gz"
            shutil.copy2(src_image, dst_image)
            
            # 复制标签
            src_label = label_files[i]
            dst_label = labels_tr / f"KiTS_{train_count:03d}.nii.gz"
            shutil.copy2(src_label, dst_label)
            
            train_count += 1
    
    # 复制测试集 (后半部分)
    test_count = 0
    for i in range(split_point, total_cases):
        if i < len(image_files) and i < len(label_files):
            # 复制图像到测试集
            src_image = image_files[i]
            dst_image = images_ts / f"KiTS_{test_count:03d}_0000.nii.gz"
            shutil.copy2(src_image, dst_image)
            
            # 复制标签到测试集 (用于后续评估)
            src_label = label_files[i]
            dst_label = labels_ts / f"KiTS_{test_count:03d}.nii.gz"
            shutil.copy2(src_label, dst_label)
            
            test_count += 1
    
    # 读取原始 dataset.json
    original_json_path = source_path / "dataset.json"
    if original_json_path.exists():
        with open(original_json_path, 'r') as f:
            original_json = json.load(f)
    else:
        # 如果没有原始json，创建默认的
        original_json = {
            "channel_names": {"0": "CT"},
            "labels": {
                "background": 0,
                "kidney": 1,
                "tumor": 2
            },
            "file_ending": ".nii.gz"
        }
    
    # 创建新的 dataset.json
    new_dataset_json = original_json.copy()
    new_dataset_json.update({
        "name": new_dataset_name,
        "description": "KiTS 2023 with custom train/test split - first half for training, second half for testing",
        "numTraining": train_count,
        "numTest": test_count,
        "training_cases": [f"KiTS_{i:03d}" for i in range(train_count)],
        "test_cases": [f"KiTS_{i:03d}" for i in range(test_count)]
    })
    
    # 保存 dataset.json
    with open(target_path / "dataset.json", 'w') as f:
        json.dump(new_dataset_json, f, indent=2)
    
    print(f"\n数据集创建完成!")
    print(f"训练样本数: {train_count}")
    print(f"测试样本数: {test_count}")
    print(f"数据集路径: {target_path}")
    
    return target_path, train_count, test_count

def create_single_fold_splits(dataset_path, output_splits_path):
    """
    为自定义数据集创建单fold的splits文件
    这样nnU-Net就不会进行交叉验证，而是使用我们指定的训练/验证划分
    """
    
    splits_path = Path(output_splits_path)
    splits_path.mkdir(parents=True, exist_ok=True)
    
    # 读取数据集信息
    dataset_json_path = Path(dataset_path) / "dataset.json"
    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)
    
    num_training = dataset_info['numTraining']
    
    # 创建训练案例列表
    train_cases = [f"KiTS_{i:03d}" for i in range(num_training)]
    
    # 简单划分：90%训练，10%验证
    val_split = int(0.9 * len(train_cases))
    train_set = train_cases[:val_split]
    val_set = train_cases[val_split:]
    
    # 创建splits文件
    splits = [
        {
            'train': train_set,
            'val': val_set
        }
    ]
    
    splits_file = splits_path / "splits_final.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"创建splits文件: {splits_file}")
    print(f"训练案例: {len(train_set)}")
    print(f"验证案例: {len(val_set)}")
    
    return splits_file

def modify_existing_dataset_for_custom_split(dataset_path, test_ratio=0.5):
    """
    修改现有数据集，将部分训练数据移动到测试集
    
    Args:
        dataset_path: 数据集路径
        test_ratio: 测试集比例 (0.5 = 后半部分作为测试集)
    """
    
    dataset_path = Path(dataset_path)
    
    # 获取所有训练文件
    images_tr = dataset_path / "imagesTr"
    labels_tr = dataset_path / "labelsTr"
    images_ts = dataset_path / "imagesTs"
    labels_ts = dataset_path / "labelsTs"
    
    # 确保测试目录存在
    images_ts.mkdir(exist_ok=True)
    labels_ts.mkdir(exist_ok=True)
    
    # 获取所有文件
    image_files = sorted(list(images_tr.glob("*.nii.gz")))
    label_files = sorted(list(labels_tr.glob("*.nii.gz")))
    
    total_cases = len(image_files)
    split_point = int(total_cases * (1 - test_ratio))
    
    print(f"重新划分数据集: {dataset_path.name}")
    print(f"总案例数: {total_cases}")
    print(f"保留训练集: {split_point} 个案例")
    print(f"移动到测试集: {total_cases - split_point} 个案例")
    
    # 移动后半部分到测试集
    moved_count = 0
    for i in range(split_point, total_cases):
        if i < len(image_files) and i < len(label_files):
            # 移动图像
            old_image = image_files[i]
            new_image = images_ts / f"test_{moved_count:03d}_0000.nii.gz"
            shutil.move(str(old_image), str(new_image))
            
            # 移动标签
            old_label = label_files[i]
            new_label = labels_ts / f"test_{moved_count:03d}.nii.gz"
            shutil.move(str(old_label), str(new_label))
            
            moved_count += 1
    
    # 更新 dataset.json
    json_path = dataset_path / "dataset.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            dataset_json = json.load(f)
        
        dataset_json['numTraining'] = split_point
        dataset_json['numTest'] = moved_count
        dataset_json['description'] = dataset_json.get('description', '') + ' - Custom train/test split applied'
        
        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
    
    print(f"数据集重新划分完成!")
    print(f"新的训练样本数: {split_point}")
    print(f"新的测试样本数: {moved_count}")

def main():
    """主函数 - 提供多种划分方式"""
    
    print("=== nnU-Net 自定义训练/测试集划分工具 ===\n")
    
    # 方法1: 创建新的数据集（推荐）
    print("方法1: 创建新数据集（推荐）")
    print("这将创建一个新的Dataset221，不影响原始数据")
    
    # 设置路径
    source_dataset = "/u/home/wyou/Documents/nnUNet/nnUNetFrame/Dataset/nnUNet_raw/Dataset220_KiTS2023"
    output_path = "/u/home/wyou/Documents/nnUNet/nnUNetFrame/Dataset/nnUNet_raw"
    
    
    if os.path.exists(source_dataset):
        target_path, train_count, test_count = create_custom_split_dataset(
            source_dataset, output_path, dataset_id=221
        )
        
        # 创建自定义splits文件
        preprocessed_path = f"nnUNet_preprocessed/Dataset221_KiTS2023_CustomSplit"
        create_single_fold_splits(target_path, preprocessed_path)
        
        print(f"\n下一步操作:")
        print(f"1. 预处理新数据集:")
        print(f"   nnUNetv2_plan_and_preprocess -d 221 --verify_dataset_integrity")
        print(f"\n2. 训练模型:")
        print(f"   nnUNetv2_train 221 3d_fullres 0")
        print(f"\n3. 在测试集上评估:")
        print(f"   nnUNetv2_predict -i {target_path}/imagesTs -o predictions -d 221 -c 3d_fullres -f 0")
        
    else:
        print(f"源数据集不存在: {source_dataset}")
        
        print(f"\n方法2: 修改现有数据集")
        print("直接修改Dataset220，将后半部分移动到测试集")
        
        existing_dataset = "nnUNet_raw/Dataset220_KiTS2023"
        if os.path.exists(existing_dataset):
            print(f"是否要修改现有数据集? 这将直接修改 {existing_dataset}")
            # modify_existing_dataset_for_custom_split(existing_dataset, test_ratio=0.5)
        else:
            print(f"数据集不存在: {existing_dataset}")

if __name__ == "__main__":
    main()