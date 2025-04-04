import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.cldice_loss import SoftclDiceLoss, SoftclDiceLoss_ori
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

class DC_and_CE_and_CLDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cldc_kwargs, weight_ce=2, weight_dice=1, weight_cldice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_CLDC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.ignore_label = ignore_label

        self.skeletonize_flag = cldc_kwargs.pop('skeletonize_flag')
        self.original_algo = cldc_kwargs.pop('original_algo')

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        if self.original_algo:
            if self.skeletonize_flag not in ['topo', 'morpho']:
                raise ValueError(f"Invalid skeletonize_flag '{self.skeletonize_flag}'. "
                            "Must be 'topo' or 'morpho'.")
            self.cldice = SoftclDiceLoss_ori(**cldc_kwargs)
        else:
            self.cldice = SoftclDiceLoss(**cldc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        cldice_loss = self.cldice(net_output, target, skeletonize_flag=self.skeletonize_flag) if self.weight_cldice != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cldice * cldice_loss
        return result

