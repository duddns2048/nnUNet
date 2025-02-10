import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss, soft_dice_cldice, SoftSkeletonize, soft_dice
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

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
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class soft_dice_cldice(nn.Module):
    def __init__(self, do_bg: bool = True, iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.do_bg = do_bg
        
        self.apply_nonlin = softmax_helper_dim1

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape
        
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            
        if not self.do_bg:
            x = x[:, 1:]
            
        x_mask  = (x > 0.5).float()
        dice = soft_dice(x, y)
        x_skel = self.soft_skeletonize(x_mask)
        y_skel = self.soft_skeletonize(y)
        tprec = (torch.sum(torch.multiply(x_skel, y))+self.smooth)/(torch.sum(x_skel)+self.smooth)    
        tsens = (torch.sum(torch.multiply(y_skel, x))+self.smooth)/(torch.sum(y_skel)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice


# Dice +clCE
class dice_clCE_loss(nn.Module):
    def __init__(self, do_bg, iter_=10, smooth=1.0, weight_clCE=1):
        super(dice_clCE_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.weight_clCE = weight_clCE
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        
        self.do_bg = do_bg
        self.apply_nonlin = softmax_helper_dim1

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: # def forward(y_true, y_pred):
        y_onehot = torch.zeros(x.shape, device=x.device)
        y_onehot.scatter_(1, y.long(), 1) # nnUNet-training-loss-dice.py

        cross_ent = torch.nn.functional.cross_entropy(x, y_onehot, reduction="none")
        x = x.softmax(dim=1)

        dice = soft_dice(y_onehot, x,)

        skel_pred = self.soft_skeletonize(x)
        skel_true = self.soft_skeletonize(y_onehot)
        tprec = torch.mul(cross_ent, skel_true[:,1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:,1]).mean()
        cl_ce = (tprec+tsens)
        result = (1.0 - self.weight_clCE) * dice + self.weight_clCE * cl_ce
        return result


class CE_cldice_loss(nn.Module):
    def __init__(self, ce_kwargs, iter_=10, smooth=1.0, weight_cldice=1, ignore_label=None):
        super(CE_cldice_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_cldice = weight_cldice
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None) else 0

        target_oh = torch.zeros(net_output.shape, device=net_output.device)
        target_oh.scatter_(1, target.long(), 1) # nnUNet-training-loss-dice.py

        net_output=net_output.softmax(dim=1)
        skel_true = SoftSkeletonize(target_oh, self.iter_)
        skel_pred = SoftSkeletonize(net_output, self.iter_)
        #tprec = (torch.sum(torch.multiply(skel_pred, target[:, 0])[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tprec = (torch.sum(torch.multiply(skel_pred, target_oh)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, net_output)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.0- 2.0*(tprec*tsens)/(tprec+tsens)

        ##Total loss computation##
        result = (1.0 -self.weight_cldice) * ce_loss + self.weight_cldice * cl_dice
        return result

class CE_clCE_loss(nn.Module):
    def __init__(self, ce_kwargs, iter_=10, weight_clCE=1):
        super(CE_clCE_loss, self).__init__()
        self.iter = iter_
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.weight_clCE = weight_clCE

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: # def forward(y_true, y_pred):
        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true.long(), 1) # nnUNet-training-loss-dice.py

        ce_loss = self.ce(y_pred, y_true[:, 0])
        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)
        skel_pred = SoftSkeletonize(y_pred, self.iter)
        skel_true = SoftSkeletonize(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:,1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:,1]).mean()
        cl_ce = (tprec+tsens)
        result = (1.0 - self.weight_clCE) * ce_loss + self.weight_clCE * cl_ce
        return result

class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

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
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
