import numpy as np
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.compound_cldice_loss import DC_and_CE_and_CLDC_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    
class nnUNetTrainer_topo_CLDC(nnUNetTrainer):

    def _build_loss(self):
        lambda_cldice = 1.0
        lambda_dice = 1.0
        lambda_ce = lambda_dice + lambda_cldice

        loss = DC_and_CE_and_CLDC_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
                                       {},
                                       {'iter_': 10, 'smooth': 1e-3, 't_skeletonize_flag': True},
                                    weight_ce=lambda_ce, 
                                    weight_dice=lambda_dice, 
                                    weight_cldice=lambda_cldice, 
                                    ignore_label=self.label_manager.ignore_label, 
                                    dice_class=MemoryEfficientSoftDiceLoss)

        self.print_to_log_file("lambda_cldice: %s" % str(lambda_cldice))
        self.print_to_log_file("lambda_dice: %s" % str(lambda_dice))
        self.print_to_log_file("lambda_ce: %s" % str(lambda_ce))

        return loss