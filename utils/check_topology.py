import nibabel as nib
import numpy as np
from scipy.ndimage import label
import networkx

def analyze_3d_mask(mask_path, target_label):
    # Load the NIfTI file
    nii = nib.load(mask_path)
    mask = nii.get_fdata()
    
    # Extract the target label
    binary_mask = (mask == target_label).astype(np.uint8)
    
    # Find connected components using 26-neighborhood for 3D
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_array, num_components = label(binary_mask, structure)
    
    print(f"Target label {target_label} forms {num_components} connected components.")
    
    # Construct graph for loop detection
    G = networkx.Graph()
    shape = binary_mask.shape
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if labeled_array[x, y, z] > 0:
                    node = (x, y, z)
                    G.add_node(node)
                    
                    # Check 6-connectivity (adjacent voxels)
                    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                            if labeled_array[nx, ny, nz] == labeled_array[x, y, z]:
                                G.add_edge((x, y, z), (nx, ny, nz))
    
    num_loops = len(list(networkx.cycle_basis(G)))
    print(f"Target label {target_label} contains {num_loops} loops.")

# Example usage
mask_path = "../Datasets/datas/nnUNet_raw/Dataset700_Parse22/labelsTr/Parse_005.nii.gz"
target_label = 1  # Adjust based on the label of interest
analyze_3d_mask(mask_path, target_label)
