# tmp environment variable
# export nnUNet_raw="/home/user/Datasets/datas/nnUNet_raw"
# export nnUNet_preprocessed="/home/user/Datasets/datas/nnUNet_preprocessed"
# export nnUNet_results="/home/user/Datasets/datas/nnUNet_results"

# preprocessing
# nnUNetv2_plan_and_preprocess -d 540 --verify_dataset_integrity

# train
nnUNetv2_train 540 3d_fullres 0 -exp_name base -save_every 50 -num_epochs 1000

nnUNetv2_train 540 3d_fullres 0 -tr nnUNetTrainer_morpho_CLDC1 -exp_name 2CE_1DC_1morpho-cldice1 -save_every 50 -num_epochs 1000

nnUNetv2_train 540 3d_fullres 0 -tr nnUNetTrainer_topo_CLDC -exp_name 2CE_1DC_1topo-cldice -save_every 50 -num_epochs 1000