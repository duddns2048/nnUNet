# tmp environment variable
# export nnUNet_raw="/home/user/Datasets/datas/nnUNet_raw"
# export nnUNet_preprocessed="/home/user/Datasets/datas/nnUNet_preprocessed"
# export nnUNet_results="/home/user/Datasets/datas/nnUNet_results"

# preprocessing
# nnUNetv2_plan_and_preprocess -d 600 --verify_dataset_integrity

# train
# nnUNetv2_train 600 2d 1 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring -save_every 1 -num_epochs 30 2>&1 | tee ./log/drive_cbdice_$(date +%Y%m%d_%H%M%S).log
nnUNetv2_train 600 2d 1 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring -save_every 1 -num_epochs 30
nnUNetv2_train 600 2d 1 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall -save_every 1 -num_epochs 30

# valid
# for i in {1..30}
# do
#     nnUNetv2_train 600 2d 1 -exp_name base_noDeepsup --val --npz -validation_ckpt $i
# done
