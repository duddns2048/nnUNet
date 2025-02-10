import os
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], validation_output_folder: str,
                                  save_probabilities: bool = False,
                                  k:str='',
                                  validation_ckpt:str=None):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(validation_output_folder+'/probability/' +k+ '_'+str(validation_ckpt)+'.npz', probabilities=probabilities_final)
        save_probability_heatmap(probabilities_final[1], validation_output_folder+'/heatmap/'+k+'_'+str(validation_ckpt)+'_heatmap.png')
        save_probability_contour(probabilities_final[1], validation_output_folder+'/contour/'+k+'_'+str(validation_ckpt)+'_contour.png')
        save_segmentation_mask(segmentation_final, validation_output_folder+'/mask/'+k+'_'+str(validation_ckpt)+'_mask.png')
        save_pickle(properties_dict, validation_output_folder + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, validation_output_folder + f'/{k}' + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    # # needed for cascade
    # if isinstance(predicted, str):
    #     assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
    #                               "isfile(segmentation_softmax) must be True"
    #     del_file = deepcopy(predicted)
    #     predicted = np.load(predicted)
    #     os.remove(del_file)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
    torch.set_num_threads(old_threads)

########### my codes ############
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
def save_probability_heatmap(input_prob_map: np.ndarray, filename: str = "probability_heatmap.png"):
    """
    확률 맵을 히트맵 형태로 PNG 파일로 저장하는 함수.

    :param input_prob_map: 확률 맵 (2D numpy array)
    :param filename: 저장할 파일명
    """
    prob_map = copy.deepcopy(input_prob_map)
    prob_map = prob_map[0]
    plt.figure(figsize=(8, 6))
    sns.heatmap(prob_map, cmap="jet", cbar=True)  # 히트맵 생성 (jet 색상 사용)
    
    plt.axis("off")  # 축 제거
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)  # PNG로 저장
    plt.close()

def save_probability_contour(input_prob_map: np.ndarray, filename: str = "probability_contour.png"):
    """
    확률 맵을 등고선 형태로 저장하는 함수.

    :param input_prob_map: 확률 맵 (2D numpy array)
    :param filename: 저장할 파일명
    """
    prob_map = copy.deepcopy(input_prob_map)
    prob_map = prob_map[0]
    plt.figure(figsize=(8, 6))
    plt.contourf(prob_map, levels=20, cmap="jet")  # 등고선 플롯
    plt.colorbar()  # 색상 바 추가
    
    plt.gca().invert_yaxis()
    
    plt.axis("off")  # 축 제거
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

from PIL import Image
def save_segmentation_mask(input_seg_map: np.ndarray, filename: str = "segmentation_mask.png"):
    """
    0과 1로 이루어진 (1, 512, 512) shape의 세그멘테이션 마스크를 PNG 파일로 저장.

    :param mask: (1, 512, 512) shape을 갖는 numpy 배열 (0 또는 1 값만 포함)
    :param filename: 저장할 PNG 파일 이름
    """
    seg_map = copy.deepcopy(input_seg_map)
    seg_map = seg_map[0]
    # 0과 255 값으로 변환 (0: 배경, 255: 전경)
    mask_2d = (seg_map * 255).astype(np.uint8)

    # PNG로 저장
    image = Image.fromarray(mask_2d)
    image.save(filename)
############ my codes ############