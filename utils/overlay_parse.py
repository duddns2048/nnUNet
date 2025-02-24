import nibabel as nib
import numpy as np

def create_tp_tn_fp_fn_mask(gt_path, pred_path, output_path):
    # Load GT and prediction masks
    gt_nii = nib.load(gt_path)
    pred_nii = nib.load(pred_path)
    
    gt = gt_nii.get_fdata().astype(np.uint8)
    pred = pred_nii.get_fdata().astype(np.uint8)
    
    # Ensure the shapes match
    if gt.shape != pred.shape:
        raise ValueError("GT and Prediction masks must have the same shape")
    
    # Define TP, TN, FP, FN labels
    tp = (gt == 1) & (pred == 1)  # True Positive (label 1)
    tn = (gt == 0) & (pred == 0)  # True Negative (label 2)
    fp = (gt == 0) & (pred == 1)  # False Positive (label 3)
    fn = (gt == 1) & (pred == 0)  # False Negative (label 4)
    
    # Create labeled mask
    labeled_mask = np.zeros(gt.shape, dtype=np.uint8)
    labeled_mask[tp] = 1
    labeled_mask[tn] = 2
    labeled_mask[fp] = 3
    labeled_mask[fn] = 4
    
    # Save the labeled mask as a new NIfTI file
    new_nii = nib.Nifti1Image(labeled_mask, affine=gt_nii.affine, header=gt_nii.header)
    nib.save(new_nii, output_path)
    
    print(f"Saved TP/TN/FP/FN mask to {output_path}")

# Example usage
gt_path = "../Datasets/datas/nnUNet_raw/Dataset700_Parse22/labelsTr/Parse_005.nii.gz"
pred_path = "../Datasets/datas/nnUNet_results/Dataset700_Parse22/nnUNetTrainer__nnUNetPlans__3d_fullres_base/fold_1/validation/Parse_005.nii.gz"
output_path = "../Datasets/datas/nnUNet_results/Dataset700_Parse22/nnUNetTrainer__nnUNetPlans__3d_fullres_base/fold_1/validation/analysis.nii.gz"
create_tp_tn_fp_fn_mask(gt_path, pred_path, output_path)
