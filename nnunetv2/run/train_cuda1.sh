CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -num_epochs 30 -save_every 1 -loss base
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 1
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 3
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 6
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 9
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 12
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 14
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 16
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 18
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name base -val --npz -validation_ckpt 20

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -num_epochs 30 -save_every 1 -loss soft_dice_cldice -cldice_alpha 0.5
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 1
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 3
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 6
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 9
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 12
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 14
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 16
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 18
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 2d 1 -exp_name cldice_0.5 -val --npz -validation_ckpt 20

