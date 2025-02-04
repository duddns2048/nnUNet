# # CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1 -num_epochs 30 -save_every 1 -loss soft_dice_cldice -cldice_alpha 0.1
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 1 --val --npz 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 3 --val --npz 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 6 --val --npz 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 9 --val --npz 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 12 --val --npz 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 15 --val --npz 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 18 --val --npz 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 21 --val --npz 
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.1_every1 -validation_ckpt 24 --val --npz 

# # CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3 -num_epochs 30 -save_every 1 -loss soft_dice_cldice -cldice_alpha 0.3
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 1
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 3
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 6
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 9
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 12
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 15
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 18
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 21
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 2d 1 -exp_name cldice_0.3_every1 --val --npz -validation_ckpt 24


CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 610 2d 1 -exp_name base -num_epochs 100 -save_every 3 -loss base

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 610 2d 1 -exp_name cldice_0.1 -num_epochs 100 -save_every 3 -loss soft_dice_cldice -cldice_alpha 0.1

