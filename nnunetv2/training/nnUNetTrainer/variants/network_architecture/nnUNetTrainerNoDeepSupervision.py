from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
        exp_name:str='',
        save_every: int=0,
        num_epochs: int=0,
        loss:str='',
        cldice_alpha:float=0.0,
        only_run_validation:bool=False,
        enable_deep_supervision:bool=False
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device,
                         exp_name,
                         save_every,
                         num_epochs,
                         loss,
                         cldice_alpha,
                         only_run_validation,
                         enable_deep_supervision)
        self.enable_deep_supervision = False
