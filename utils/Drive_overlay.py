import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def compute_dice(gt, pred):
    intersection = np.sum(gt * pred)
    return (2. * intersection) / (np.sum(gt) + np.sum(pred) + 1e-6)

def visualize_overlay(gt_folder, pred_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract unique case IDs from pred_folder
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".npz")])
    unique_cases = set("_".join(f.split("_")[:-1]) for f in pred_files)
    
    for case_id in unique_cases:
        gt_file = f"{case_id}.tif"
        gt_path = os.path.join(gt_folder, gt_file)
        
        if not os.path.exists(gt_path):
            print(f"GT file not found: {gt_file}")
            continue
        
        # Load GT mask (binary image)
        gt = np.array(Image.open(gt_path))
        gt = (gt > 0).astype(np.uint8)  # Ensure binary values (0 or 1)
        
        # Process each epoch's prediction for this case
        case_pred_files = sorted([f for f in pred_files if f.endswith('_29.npz') and f.startswith(case_id)])
        for pred_file in case_pred_files:
            epoch = pred_file.split("_")[-1].split(".")[0]  # Extract epoch number
            
            pred_path = os.path.join(pred_folder, pred_file)
            pred_data = np.load(pred_path)["probabilities"]  # (2,1,H,W)
            pred_mask = pred_data[1, 0] > 0.5  # Threshold at 0.5 to create binary mask
            
            # Compute Dice Score
            dice_score = compute_dice(gt, pred_mask)
            
            # Compute TP, TN, FP, FN
            TP = np.sum((pred_mask == 1) & (gt == 1))
            TN = np.sum((pred_mask == 0) & (gt == 0))
            FP = np.sum((pred_mask == 1) & (gt == 0))
            FN = np.sum((pred_mask == 0) & (gt == 1))
            
            # Create an overlay image
            overlay = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
            overlay[(pred_mask == 1) & (gt == 1)] = [0, 255, 0]   # TP (Green)
            overlay[(pred_mask == 0) & (gt == 0)] = [0, 0, 0]     # TN (Black)
            overlay[(pred_mask == 1) & (gt == 0)] = [255, 0, 0]   # FP (Red)
            overlay[(pred_mask == 0) & (gt == 1)] = [255, 255, 255] # FN (White)

            output_path = os.path.join(output_folder, f"{case_id}_epoch30_overlay.png")
            
            # Plot the results
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(overlay)
            ax.set_title(f"Overlay Image - {case_id} (Epoch 30)")
            ax.axis("off")
            text = f"Dice: {dice_score:.4f}\nTP: {TP}, TN: {TN}, \nFP: {FP}, FN: {FN}"
            plt.figtext(0.5, 0.02, text, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
            plt.savefig(output_path, bbox_inches='tight', dpi=600)
            plt.close(fig)

            # overlay_img = Image.fromarray(overlay)
            # draw = ImageDraw.Draw(overlay_img)            
            # text = (f"Overlay Image - {case_id} (Epoch {epoch})\n"
            #         f"Dice: {dice_score:.4f}\n"
            #         f"TP: {TP}\nTN: {TN}\nFP: {FP}\nFN: {FN}")
            # font = ImageFont.load_default()
            # draw.text((10, 10), text, fill=(255, 255, 255), font=font)
            # overlay_img.save(output_path, dpi=(600, 600))
            
            print(f"Saved: {output_path}")

# Example usage
gt_folder = "../Datasets/datas/nnUNet_raw/Dataset600_DRIVE/labelsTr/"
pred_folder = "../Datasets/datas/nnUNet_results/Dataset600_DRIVE/nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring__nnUNetPlans__2d_CE_DC_CBDC_NoDeep_NoMirr/fold_1/validation/probability"
output_folder = "../Datasets/datas/nnUNet_results/Dataset600_DRIVE/nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_NoMirroring__nnUNetPlans__2d_CE_DC_CBDC_NoDeep_NoMirr/fold_1/validation/analysis"
visualize_overlay(gt_folder, pred_folder, output_folder)