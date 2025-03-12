# tmp environment variable
# export nnUNet_raw="/home/user/Datasets/datas/nnUNet_raw"
# export nnUNet_preprocessed="/home/user/Datasets/datas/nnUNet_preprocessed"
# export nnUNet_results="/home/user/Datasets/datas/nnUNet_results"

# preprocessing
# nnUNetv2_plan_and_preprocess -d 700 --verify_dataset_integrity

# train
# nnUNetv2_train 700 3d_fullres 3 -exp_name base -save_every 50 -num_epochs 1000 -loss base
# nnUNetv2_train 700 3d_fullres 3 -exp_name cldice_0.5 -save_every 50 -num_epochs 1000 -loss soft_dice_cldice -cldice_alpha 0.5
# nnUNetv2_train 700 3d_fullres 0 -tr nnUNetTrainerDiceLoss -exp_name only_dice -save_every 50 -num_epochs 1000
# nnUNetv2_train 700 3d_fullres 3 -tr nnUNetTrainerDiceLoss -exp_name only_dice -save_every 50 -num_epochs 1000
# nnUNetv2_train 700 3d_fullres 1 -tr nnUNetTrainerCELoss -exp_name only_ce -save_every 50 -num_epochs 1000

nnUNetv2_train 700 3d_fullres 0 -tr nnUNetTrainer_CE_DC_CLDC -exp_name 2CE_1DC_1cldice -save_every 1 -num_epochs 30

# valid
# nnUNetv2_train 700 3d_fullres 3 -tr nnUNetTrainerDiceLoss -exp_name only_dice --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 3 -tr nnUNetTrainerDiceLoss -exp_name only_dice --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 1 -exp_name base --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 1 -exp_name cldice_0.5 --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 0 -tr nnUNetTrainerCELoss -exp_name only_ce --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 1 -tr nnUNetTrainerCELoss -exp_name only_ce --val --npz -validation_ckpt 1000


# nnUNetv2_train 700 3d_fullres 0 -exp_name base --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 1 -exp_name base --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 2 -exp_name base --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 3 -exp_name base --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 4 -exp_name base --val --npz -validation_ckpt 1000

# nnUNetv2_train 700 3d_fullres 0 -exp_name cldice_0.5 --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 1 -exp_name cldice_0.5 --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 2 -exp_name cldice_0.5 --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 3 -exp_name cldice_0.5 --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 4 -exp_name cldice_0.5 --val --npz -validation_ckpt 1000

# nnUNetv2_train 700 3d_fullres 0 -tr nnUNetTrainerSkeletonRecall -exp_name skel_recall --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 1 -tr nnUNetTrainerSkeletonRecall -exp_name skel_recall --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 2 -tr nnUNetTrainerSkeletonRecall -exp_name skel_recall --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 3 -tr nnUNetTrainerSkeletonRecall -exp_name skel_recall --val --npz -validation_ckpt 1000
# nnUNetv2_train 700 3d_fullres 4 -tr nnUNetTrainerSkeletonRecall -exp_name skel_recall --val --npz -validation_ckpt 1000