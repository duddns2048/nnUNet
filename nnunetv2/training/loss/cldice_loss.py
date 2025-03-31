import torch
from nnunetv2.training.loss.skeletonize import Skeletonize
from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np


def rand_sampling_skel(binary_mask, sampling_ratio=0.5):
    assert 0 <= sampling_ratio <= 1, "sampling_ratio는 0과 1 사이 값이어야 합니다."
    assert binary_mask.dim() in (4, 5), "binary_mask는 (B,C,H,W) 또는 (B,C,H,W,D) 형태여야 합니다."

    output_tensor = torch.zeros_like(binary_mask)

    for b in range(binary_mask.size(0)):
        for c in range(binary_mask.size(1)):
            current_mask = binary_mask[b, c]
            ones_indices = torch.nonzero(current_mask == 1, as_tuple=False)

            num_ones = ones_indices.size(0)
            num_samples = int(num_ones * sampling_ratio)

            if num_samples == 0:
                continue

            sampled_indices = ones_indices[torch.randperm(num_ones)[:num_samples]]

            if current_mask.dim() == 2:
                output_tensor[b, c, sampled_indices[:, 0], sampled_indices[:, 1]] = 1
            elif current_mask.dim() == 3:
                output_tensor[b, c, sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]] = 1

    return output_tensor

def ski_skel(binary_mask):
    """
    (B,C,H,W) 또는 (B,C,H,W,D) 형태의 입력 텐서에 대해 skeletonize 또는 skeletonize_3d를 적용하는 함수

    Parameters:
        binary_mask (torch.Tensor): 입력 텐서 (B,C,H,W) or (B,C,H,W,D)

    Returns:
        torch.Tensor: Skeletonized된 출력 텐서 (입력과 동일한 shape)
    """
    assert binary_mask.dim() in (4, 5), "binary_mask는 (B,C,H,W) 또는 (B,C,H,W,D) 형태여야 합니다."

    skel_result_list = []

    for b in range(binary_mask.size(0)):
        channel_result_list = []
        for c in range(binary_mask.size(1)):
            sample = binary_mask[b, c]
            sample_np = sample.cpu().numpy().astype(bool)

            if sample_np.ndim == 2:
                skel_np = skeletonize(sample_np)
            elif sample_np.ndim == 3:
                skel_np = skeletonize_3d(sample_np)
            else:
                raise ValueError(f"Unsupported number of dimensions: {sample_np.ndim}")

            skel_tensor = torch.from_numpy(skel_np.astype(np.float32))
            channel_result_list.append(skel_tensor)

        skel_result_list.append(torch.stack(channel_result_list, dim=0))

    skel_result = torch.stack(skel_result_list, dim=0)

    return skel_result

class SoftclDiceLoss(torch.nn.Module):
    def __init__(self, iter_=20, smooth = 1.):
        super(SoftclDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

        # skimage skel2d, skel3d
        self.ski_skeletonize = ski_skel

        # random sampling 
        self.rand_skeletonize = rand_sampling_skel

    def forward(self, y_pred, y_true, skeletonize_flag=None):
        """
        Forward pass for the loss function.
        
        Args:
            y_pred (torch.Tensor): Network output with shape (b, c, x, y(, z)).
            y_true (torch.Tensor): Ground truth labels with shape (b, 1, x, y(, z)). No one-hot encoding required.
            skeletonize_flag (bool, optional): Enable Topology-preserving skeletonization. Defaults to False.
        """
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1] # 여기서 C차원 삭제

        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
        
            if skeletonize_flag=='topo':
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            elif skeletonize_flag=='morpho':
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            elif skeletonize_flag=='ski':
                skel_pred_hard = self.ski_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.ski_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            elif skeletonize_flag=='rand':
                skel_pred_hard = self.rand_skeletonize(y_pred_hard.unsqueeze(1), 0.5).squeeze(1)
                skel_true = self.rand_skeletonize(y_true.unsqueeze(1), 0.5).squeeze(1)

        skel_pred_prob = skel_pred_hard * y_pred_prob # here is the point of algo 1

        tprec = (torch.sum(torch.multiply(skel_pred_prob, y_true))+self.smooth)/(torch.sum(skel_pred_prob)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred_prob))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice_loss = 1 - 2.0 * (tprec*tsens)/(tprec+tsens)

        return cl_dice_loss
    
class SoftclDiceLoss_ori(torch.nn.Module):
    def __init__(self, iter_=20, smooth = 1.):
        super(SoftclDiceLoss_ori, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, skeletonize_flag=True):
        """
        Forward pass for the loss function.
        
        Args:
            y_pred (torch.Tensor): Network output with shape (b, c, x, y(, z)).
            y_true (torch.Tensor): Ground truth labels with shape (b, 1, x, y(, z)). No one-hot encoding required.
            skeletonize_flag (bool, optional): Enable Topology-preserving skeletonization. Defaults to False.
        """
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]
        y_pred_prob = y_pred_prob.unsqueeze(1)

        y_true = torch.where(y_true > 0, 1, 0).float()
        
        if skeletonize_flag=='topo':
            skel_pred_hard = self.t_skeletonize(y_pred_prob)
            skel_true = self.t_skeletonize(y_true)
        elif skeletonize_flag=='morpho':
            skel_pred_hard = self.m_skeletonize(y_pred_prob)
            skel_true = self.m_skeletonize(y_true)

        tprec = (torch.sum(torch.multiply(skel_pred_hard, y_true))+self.smooth)/(torch.sum(skel_pred_hard)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred_prob))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice_loss = 1 - 2.0 * (tprec*tsens)/(tprec+tsens)

        return cl_dice_loss



class SoftclDiceLoss3(torch.nn.Module): # modified soft_dice_cldice from cldice official github
    def __init__(self, iter_=20, smooth = 1., exclude_background=True):
        super(SoftclDiceLoss3, self).__init__()
        self.smooth = smooth
        self.exclude_background = exclude_background

        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, skeletonize_flag=True):
        y_prob = torch.softmax(y_pred, 1)
        # y_pred.shape= (B,C,H,W,(D))
        if self.exclude_background:
            y_true = y_true[:, 1:]
            y_prob = y_prob[:, 1:] # multiclass인 경우우one-hot 해줘야되는거아님? 마지막에 class별 cldice 평균도 해줘야되는거 아닌가? 아님걍 FG로 해석하는게 맞나?

        if skeletonize_flag:
            skel_pred_hard = self.t_skeletonize(y_prob)
            skel_true = self.t_skeletonize(y_true)
        else:
            skel_pred_hard = self.m_skeletonize(y_prob)
            skel_true = self.m_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred_hard, y_true))+self.smooth)/(torch.sum(skel_pred_hard)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_prob))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        return 1.- 2.0*(tprec*tsens)/(tprec+tsens)