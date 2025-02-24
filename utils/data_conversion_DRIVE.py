import os
import shutil

import os
import numpy as np
from PIL import Image



# 대상 폴더가 없으면 생성


# 파일 리스트 가져오기
def rename_DRIVE_training_data(source_dir, destination_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith("_training.tif"):
            # 기존 파일명에서 숫자 부분 추출
            num_part = filename.split("_")[0]  # 예: "40" 추출

            # 새로운 파일명 생성 (예: DRIVE_040_0000.tif)
            new_filename = f"DRIVE_{int(num_part):03d}_0000.tif"

            # 원본 파일 경로 및 대상 파일 경로 설정
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(destination_dir, new_filename)

            # 파일 복사
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {filename} -> {new_filename}")

source_dir = "../Datasets/datasets (2)/training/images"  # 기존 폴더 경로
destination_dir = "../Datasets/datas/nnUNet_raw/Dataset600_DRIVE/imagesTr"  # 복사할 폴더 경로
# os.makedirs(destination_dir, exist_ok=True)
# rename_DRIVE_training_data(source_dir, destination_dir)

########################################################################################################################################

def convert_gif_to_binary_tif(source_folder):
    """
    source_folder 내의 모든 .gif 파일을 이진화(0 또는 1)한 후,
    raw 폴더를 생성하여 .tif 형식으로 저장하는 함수.
    
    Args:
        source_folder (str): 변환할 파일이 있는 디렉토리 경로
    """
    # raw 폴더 생성
    raw_folder = os.path.join(source_folder, "raw")
    os.makedirs(raw_folder, exist_ok=True)

    # 파일 리스트 가져오기
    for filename in os.listdir(source_folder):
        if filename.endswith("_manual1.gif"):
            # 파일 경로
            file_path = os.path.join(source_folder, filename)

            # 이미지 열기
            img = Image.open(file_path).convert("L")  # 흑백 변환 (L 모드)

            # NumPy 배열로 변환
            img_array = np.array(img)

            # 255 -> 1, 그 외 값 -> 0
            binary_array = (img_array == 255).astype(np.uint8)

            # 새로운 파일명 생성 (예: 21_manual1.gif -> 21_manual1.tif)
            new_filename = os.path.splitext(filename)[0] + ".tif"
            new_file_path = os.path.join(raw_folder, new_filename)

            # 변환된 데이터를 tif로 저장
            Image.fromarray(binary_array).save(new_file_path)

            print(f"Converted: {filename} -> {new_filename}")

    print("✅ 모든 파일이 성공적으로 변환되었습니다.")

# 사용 예시
source_folder = "../Datasets/datasets (2)/training/1st_manual"  # 변환할 폴더 경로 설정
# convert_gif_to_binary_tif(source_folder)

########################################################################################################################################
import os
import shutil

def rename_DRIVE_gt_data(source_folder, destination_folder):
    """
    source_folder 내의 모든 .tif 파일을 새로운 이름 형식으로 변경하여
    destination_folder로 복사하는 함수.

    Args:
        source_folder (str): 원본 파일이 있는 디렉토리 경로
        destination_folder (str): 변경된 파일을 저장할 디렉토리 경로
    """
    # 대상 폴더가 없으면 생성
    os.makedirs(destination_folder, exist_ok=True)

    # 파일 리스트 가져오기
    for filename in os.listdir(source_folder):
        if filename.endswith("_manual1.tif"):
            # 기존 파일명에서 숫자 부분 추출
            num_part = filename.split("_")[0]  # 예: "36" 추출

            # 새로운 파일명 생성 (예: DRIVE_036.tif)
            new_filename = f"DRIVE_{int(num_part):03d}.tif"

            # 원본 파일 경로 및 대상 파일 경로 설정
            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(destination_folder, new_filename)

            # 파일 복사
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {filename} -> {new_filename}")

    print("✅ 모든 파일이 성공적으로 복사되었습니다.")

# 사용 예시
source_folder = "../Datasets/datasets (2)/training/1st_manual/raw/"  # 원본 폴더 경로
destination_folder = "../Datasets/datas/nnUNet_raw/Dataset600_DRIVE/labelsTr"  # 복사할 폴더 경로
# rename_DRIVE_gt_data(source_folder, destination_folder)
