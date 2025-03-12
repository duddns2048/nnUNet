## tmp environment variable
# export nnUNet_raw="/home/user/Datasets/datas/nnUNet_raw"
# export nnUNet_preprocessed="/home/user/Datasets/datas/nnUNet_preprocessed"
# export nnUNet_results="/home/user/Datasets/datas/nnUNet_results"

## preprocessing
# nnUNetv2_plan_and_preprocess -d 600 --verify_dataset_integrity

## train
# nnUNetv2_train 600 2d 0 -exp_name tmp -save_every 1 -num_epochs 30 -loss base
# nnUNetv2_train 600 2d 1 -exp_name base -save_every 1 -num_epochs 30 -loss base
# nnUNetv2_train 600 2d 2 -exp_name base -save_every 1 -num_epochs 30 -loss base
# nnUNetv2_train 600 2d 3 -exp_name base -save_every 1 -num_epochs 30 -loss base
# nnUNetv2_train 600 2d 4 -exp_name base -save_every 1 -num_epochs 30 -loss base

# nnUNetv2_train 600 2d 0 -exp_name only_CE -tr nnUNetTrainerCELoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 1 -exp_name only_CE -tr nnUNetTrainerCELoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 2 -exp_name only_CE -tr nnUNetTrainerCELoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 3 -exp_name only_CE -tr nnUNetTrainerCELoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 4 -exp_name only_CE -tr nnUNetTrainerCELoss -save_every 1 -num_epochs 30

# nnUNetv2_train 600 2d 0 -exp_name only_Dice -tr nnUNetTrainerDiceLoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 1 -exp_name only_Dice -tr nnUNetTrainerDiceLoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 2 -exp_name only_Dice -tr nnUNetTrainerDiceLoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 3 -exp_name only_Dice -tr nnUNetTrainerDiceLoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 4 -exp_name only_Dice -tr nnUNetTrainerDiceLoss -save_every 1 -num_epochs 30

# nnUNetv2_train 600 2d 0 -exp_name soft_dice_cldice_0.5 -save_every 1 -num_epochs 30 -loss soft_dice_cldice -cldice_alpha 0.5
# nnUNetv2_train 600 2d 1 -exp_name soft_dice_cldice_0.5 -save_every 1 -num_epochs 30 -loss soft_dice_cldice -cldice_alpha 0.5
# nnUNetv2_train 600 2d 2 -exp_name soft_dice_cldice_0.5 -save_every 1 -num_epochs 30 -loss soft_dice_cldice -cldice_alpha 0.5
# nnUNetv2_train 600 2d 3 -exp_name soft_dice_cldice_0.5 -save_every 1 -num_epochs 30 -loss soft_dice_cldice -cldice_alpha 0.5
# nnUNetv2_train 600 2d 4 -exp_name soft_dice_cldice_0.5 -save_every 1 -num_epochs 30 -loss soft_dice_cldice -cldice_alpha 0.5

nnUNetv2_train 600 2d 0 -tr nnUNetTrainer_CE_DC_CLDC -exp_name 2CE_1DC_1cldice -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 1 -tr nnUNetTrainer_CE_DC_CLDC -exp_name 2CE_1DC_1cldice -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 2 -tr nnUNetTrainer_CE_DC_CLDC -exp_name 2CE_1DC_1cldice -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 3 -tr nnUNetTrainer_CE_DC_CLDC -exp_name 2CE_1DC_1cldice -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 4 -tr nnUNetTrainer_CE_DC_CLDC -exp_name 2CE_1DC_1cldice -save_every 1 -num_epochs 30

# nnUNetv2_train 600 2d 0 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 1 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 2 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 3 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 4 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring -save_every 1 -num_epochs 30

# nnUNetv2_train 600 2d 0 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 1 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 2 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 3 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 4 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall -save_every 1 -num_epochs 30



# nnUNetv2_train 600 2d 1 -exp_name only_CE -tr nnUNetTrainerCELoss -save_every 1 -num_epochs 30
# nnUNetv2_train 600 2d 1 -exp_name only_Dice -tr nnUNetTrainerDiceLoss -save_every 1 -num_epochs 30

## valid
# nnUNetv2_train 600 2d 1 -exp_name only_CE -tr nnUNetTrainerCELoss --val --npz -validation_ckpt 30

# nnUNetv2_train 600 2d 0 -exp_name base -save_every 1 --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 1 -exp_name base -save_every 1 --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 2 -exp_name base -save_every 1 --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 3 -exp_name base -save_every 1 --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 4 -exp_name base -save_every 1 --val --npz -validation_ckpt 30 

# nnUNetv2_train 600 2d 0 -exp_name only_CE -tr nnUNetTrainerCELoss --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 1 -exp_name only_CE -tr nnUNetTrainerCELoss --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 2 -exp_name only_CE -tr nnUNetTrainerCELoss --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 3 -exp_name only_CE -tr nnUNetTrainerCELoss --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 4 -exp_name only_CE -tr nnUNetTrainerCELoss --val --npz -validation_ckpt 30 

# nnUNetv2_train 600 2d 0 -exp_name only_Dice -tr nnUNetTrainerDiceLoss --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 1 -exp_name only_Dice -tr nnUNetTrainerDiceLoss --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 2 -exp_name only_Dice -tr nnUNetTrainerDiceLoss --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 3 -exp_name only_Dice -tr nnUNetTrainerDiceLoss --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 4 -exp_name only_Dice -tr nnUNetTrainerDiceLoss --val --npz -validation_ckpt 30 

# nnUNetv2_train 600 2d 0 -exp_name soft_dice_cldice_0.5 --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 1 -exp_name soft_dice_cldice_0.5 --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 2 -exp_name soft_dice_cldice_0.5 --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 3 -exp_name soft_dice_cldice_0.5 --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 4 -exp_name soft_dice_cldice_0.5 --val --npz -validation_ckpt 30 

# nnUNetv2_train 600 2d 0 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 1 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 2 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 3 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 4 -exp_name cbdice -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring --val --npz -validation_ckpt 30 

# nnUNetv2_train 600 2d 0 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 1 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 2 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 3 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall --val --npz -validation_ckpt 30 
# nnUNetv2_train 600 2d 4 -exp_name skel_recall -tr nnUNetTrainerSkeletonRecall --val --npz -validation_ckpt 30 

## find best config
# nnUNetv2_find_best_configuration 600 -c 2d -exp_name base