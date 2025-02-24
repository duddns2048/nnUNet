# tmp environment variable
# export nnUNet_raw="/home/user/Datasets/datas/nnUNet_raw"
# export nnUNet_preprocessed="/home/user/Datasets/datas/nnUNet_preprocessed"
# export nnUNet_results="/home/user/Datasets/datas/nnUNet_results"

# preprocessing
# nnUNetv2_plan_and_preprocess -d 700 --verify_dataset_integrity

# train
# nnUNetv2_train 700 3d_fullres 1 -exp_name base -save_every 10 -num_epochs 1000 -loss base > ./log/parse_base_$(date +%Y%m%d_%H%M%S).log 2>&1
nnUNetv2_train 700 3d_fullres 1 -exp_name base -save_every 10 -num_epochs 1000 -loss base 2>&1 | tee ./log/parse_base_$(date +%Y%m%d_%H%M%S).log

nnUNetv2_train 700 3d_fullres 1 -exp_name cldice_0.3 -save_every 10 -num_epochs 1000 -loss soft_dice_cldice -cldice_alpha 0.3 > ./log/parse_cldice_$(date +%Y%m%d_%H%M%S).log 2>&1


# valid
for i in {10..150..10}  # 10부터 150까지 10씩 증가
do
    nnUNetv2_train 700 3d_fullres 1 -exp_name base --val --npz -validation_ckpt $i
    nnUNetv2_train 700 3d_fullres 1 -exp_name cldice_0.3 --val --npz -validation_ckpt $i
done