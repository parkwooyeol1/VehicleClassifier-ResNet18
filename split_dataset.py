import os
import shutil
import random

# 원본 데이터 경로
DATASET_DIR = "/home/parkwooyeol/workspace/Resnet-Experiment/downloads"
# 나눌 대상 경로
OUTPUT_DIR = "/home/parkwooyeol/workspace/Resnet-Experiment/dataset"

# 비율 설정 (70% train, 15% val, 15% test)
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# 시드 고정 (재현성)
random.seed(42)

def split_dataset():
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        # 이미지 파일 리스트
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * SPLIT_RATIOS["train"])
        n_val = int(n_total * SPLIT_RATIOS["val"])
        n_test = n_total - n_train - n_val

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        # 각 split 폴더에 복사
        for split_name, split_files in splits.items():
            split_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for f in split_files:
                src = os.path.join(class_path, f)
                dst = os.path.join(split_dir, f)
                shutil.copy(src, dst)

        print(f"[+] {class_name}: train {n_train}, val {n_val}, test {n_test}")

if __name__ == "__main__":
    split_dataset()
    print("Dataset split completed!")
