# nnUNetv2_train 600 2d 1 -exp_name soft_dice_cldice_0.1 -save_every 1 -num_epochs 100 -loss soft_dice_cldice -cldice_alpha 0.1
# nnUNetv2_train 600 2d 1 -exp_name soft_dice_cldice_0.3 -save_every 1 -num_epochs 100 -loss soft_dice_cldice -cldice_alpha 0.3
# nnUNetv2_train 600 2d 1 -exp_name soft_dice_cldice_0.5 -save_every 1 -num_epochs 100 -loss soft_dice_cldice -cldice_alpha 0.5

# nnUNetv2_train 600 2d 1 -exp_name soft_dice_clCE_0.1 -save_every 1 -num_epochs 100 -loss soft_dice_clCE -cldice_alpha 0.1
# nnUNetv2_train 600 2d 1 -exp_name soft_dice_clCE_0.3 -save_every 1 -num_epochs 100 -loss soft_dice_clCE -cldice_alpha 0.3
# nnUNetv2_train 600 2d 1 -exp_name soft_dice_clCE_0.5 -save_every 1 -num_epochs 100 -loss soft_dice_clCE -cldice_alpha 0.5

# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 1
# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 2
# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 3
# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 4
# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 5
# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 6
# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 7
# nnUNetv2_train 600 2 1 -exp_name base --npz --val -validation_ckpt 8
# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 9
# nnUNetv2_train 600 2d 1 -exp_name base --npz --val -validation_ckpt 10

# for i in $(seq 1 30)
# do
#     nnUNetv2_train 600 2d 1 -tr nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring -exp_name CE_DC_CBDC_NoDeep_NoMirr --npz --val -validation_ckpt $i
# done


nnUNetv2_train 600 2d 1 -tr nnUNetTrainer -exp_name base --val -validation_ckpt 20
