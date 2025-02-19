import os
import shutil

def rename_and_copy(source_dir="train", target_dir="processed_data"):
    """
    주어진 source 디렉토리에서 image 및 label 파일을 특정 형식으로 이름 변경 후 target 디렉토리에 복사하는 함수.

    Parameters:
        source_dir (str): 원본 데이터가 있는 디렉토리 (기본값: "train").
        target_dir (str): 복사된 데이터를 저장할 디렉토리 (기본값: "processed_data").
    """
    # 새로운 디렉토리 경로 설정
    target_image_dir = os.path.join(target_dir, "imagesTr")
    target_label_dir = os.path.join(target_dir, "labelsTr")

    # 저장할 디렉토리 생성
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    # train 디렉토리 내부의 폴더 순회
    for patient_id in os.listdir(source_dir):
        patient_path = os.path.join(source_dir, patient_id)

        if not os.path.isdir(patient_path):
            continue  # 폴더가 아닌 경우 건너뜀

        # 원본 파일 경로 설정
        image_path = os.path.join(patient_path, "image", f"{patient_id}.nii.gz")
        label_path = os.path.join(patient_path, "label", f"{patient_id}.nii.gz")

        if os.path.exists(image_path) or os.path.exists(label_path):
            # 숫자 부분만 추출 (PA000005 → 005)
            number = patient_id[2:].lstrip("0").zfill(3)  # 앞의 'PA' 제거, 앞의 0 제거 후 3자리 숫자로 포맷

            # 새로운 파일명 설정
            new_image_name = f"Parse_{number}_0000.nii.gz"
            new_label_name = f"Parse_{number}.nii.gz"

            # 대상 파일 경로 설정
            target_image_path = os.path.join(target_image_dir, new_image_name)
            target_label_path = os.path.join(target_label_dir, new_label_name)

            # 이미지 파일 복사
            if os.path.exists(image_path):
                shutil.copy(image_path, target_image_path)
                print(f"Copied: {image_path} -> {target_image_path}")

            # 라벨 파일 복사
            if os.path.exists(label_path):
                shutil.copy(label_path, target_label_path)
                print(f"Copied: {label_path} -> {target_label_path}")

    print("✅ 모든 파일 복사 완료!")

# 실행 예시
if __name__ == "__main__":
    rename_and_copy("../Datasets/Parse2022/train", "../Datasets/datas/nnUNet_raw/Dataset700_Parse22/")
