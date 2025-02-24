import os
import shutil
import re

# 원본 디렉토리와 복사할 대상 디렉토리 설정
source_dir = "/home/kimyw/github/Datasets/TREND/4_HEALTHY OLD RAW_GOOD QUALITY"  # 원본 폴더 경로 (변경 가능)
target_dir = "/home/kimyw/github/Datasets/datas/nnUNet_raw/Dataset610_TREND/imagesTr"  # 복사할 폴더 경로 (변경 가능)

# 대상 디렉토리가 없으면 생성
os.makedirs(target_dir, exist_ok=True)

# 이미지를 변환하는 함수
def rename_and_copy_files(source_dir, target_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".tif"):
            # 파일명에서 L/R 구분 및 숫자 또는 문자열 추출
            match = re.match(r"([A-Za-z]*)(\d+|N\d+C)_([LR])\.tif", filename)
            if match:
                prefix, num, side = match.groups()

                # 숫자인 경우 3자리로 패딩 (ex: 1 → 001, 10 → 010)
                if num.isdigit():
                    num = num.zfill(3)  # 숫자 패딩
                else:
                    num = num  # N10C 같은 경우 그대로 사용

                # 새로운 파일명 생성
                new_filename = f"trend_{side}_{num}_0000.tif"

                # 파일 복사
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(target_dir, new_filename)
                # shutil.copy2(src_path, dst_path)

                print(f"Copied: {filename} -> {new_filename}")

# 실행
# rename_and_copy_files(source_dir, target_dir)

import os
import shutil
import re
from PIL import Image

source_dir = "/home/kimyw/github/Datasets/TREND/2_HEALTHY YOUNG SEGMENTED"  # 원본 폴더 (변경 가능)
target_dir = "/home/kimyw/github/Datasets/datas/nnUNet_raw/Dataset610_TREND/labelsTr"  # 변경된 파일을 저장할 폴더

os.makedirs(target_dir, exist_ok=True)


# Segmentation 마스크를 변환하는 함수
def rename_and_convert_segmentation(source_dir, target_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".png"):  # PNG 파일만 처리
            # 숫자 기반 파일 (ex: 10_L_SEG.png)
            match_num = re.match(r"(\d+)_([LR])_SEG\.png", filename)
            # 알파벳+숫자 기반 파일 (ex: N10C_L_SEG.png)
            match_alpha = re.match(r"([A-Za-z0-9]+)_([LR])_SEG\.png", filename)

            if match_num:  # 숫자인 경우
                num, side = match_num.groups()
                num = num.zfill(3)  # 숫자를 3자리로 패딩
            elif match_alpha:  # 알파벳+숫자 조합인 경우
                num, side = match_alpha.groups()
            else:
                continue  # 매칭되지 않으면 무시

            # 새로운 파일명 생성
            new_filename = f"trend_{side}_{num}.tif"
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, new_filename)

            # PNG → TIF 변환
            try:
                with Image.open(src_path) as img:
                    # img.save(dst_path, format="TIFF")  # TIFF 포맷으로 저장
                    pass
                print(f"Converted: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")


input_dir = "../Datasets/datas/nnUNet_raw/Dataset610_TREND/labelsTr/raw"   # 원본 TIFF 파일 경로
output_dir = "../Datasets/datas/nnUNet_raw/Dataset610_TREND/labelsTr" # 변환된 TIFF 파일 경로

import numpy as np
# 0,255인 tif를 0,1 tif로 바꾸는 함수
def convert_tiff_to_binary(input_path, output_path):
    """ 단일 TIFF 이미지를 불러와서 0 또는 1 값으로 변환 후 저장 """
    with Image.open(input_path) as img:
        img_array = np.array(img)  # NumPy 배열 변환

    # 0 또는 255 → 0 또는 1로 변환
    binary_array = (img_array // 255).astype(np.uint8)  # 255 → 1, 0 → 0

    # NumPy 배열을 PIL 이미지로 변환
    binary_image = Image.fromarray(binary_array)

    # TIFF 포맷으로 저장
    binary_image.save(output_path, format="TIFF")
    
    print(f"Converted: {input_path} → {output_path}")

def process_all_tiff_images(input_dir, output_dir):
    """ 입력 폴더 내 모든 TIFF 파일을 변환하여 저장 """
    for filename in os.listdir(input_dir):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)  # 동일한 파일명 유지
            
            convert_tiff_to_binary(input_path, output_path)


# 실행
process_all_tiff_images(input_dir, output_dir)