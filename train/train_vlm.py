# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Tuple
from PIL import Image, ImageFile
from packaging import version
import numpy as np
import cv2
import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed
import concurrent.futures

from transformers import AutoConfig
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    IMAGE_TOKEN_INDEX, TOR_TOKEN_INDEX, BOX3D_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, \
    tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord
from llava.gt_points_load_utils import calculate_frame_timestamps

from llava.model.utils import insert_tor

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

try:
    # Adjust the import path based on your project structure
    from llava.spar_utils import DRAW_FUNCTIONS

    HAS_DRAW_MARKER = True
except ImportError:
    print("Warning: draw_marker.py not found. Marker drawing will be disabled.")
    DRAW_FUNCTIONS = {}  # Define as empty dict if import fails
    HAS_DRAW_MARKER = False


def _process_spar_task(task_key, task_meta, manifest_path, data_args):
    """Processes a single SPAR task defined in the manifest."""
    annotation_rel_path = task_meta.get("annotation")
    spar_image_root_rel = task_meta.get("root")
    repeat_time = float(task_meta.get("repeat_time", 1.0))
    task_type = task_key.split("_")[1] if len(task_key.split("_")) > 1 else "unknown"

    if not annotation_rel_path:
        rank0_print(f" Warning: Skipping SPAR task {task_key} in {manifest_path} (missing 'annotation').")
        return []
    if not spar_image_root_rel:
        rank0_print(f" Warning: Skipping SPAR task {task_key} in {manifest_path} (missing 'root' in manifest).")
        return []

    # Construct absolute paths
    if not os.path.isabs(annotation_rel_path):
        # Try relative to data_args.video_folder first
        annotation_path = os.path.join(data_args.video_folder or ".", annotation_rel_path)
        # If not found, try relative to the manifest file's directory
        if not os.path.exists(annotation_path):
            manifest_dir = os.path.dirname(manifest_path)
            annotation_path_alt = os.path.join(manifest_dir, annotation_rel_path)
            if os.path.exists(annotation_path_alt):
                annotation_path = annotation_path_alt
            # Keep the original path for the warning message if neither exists
    else:
        annotation_path = annotation_rel_path

    if not os.path.isabs(spar_image_root_rel):
        # Try relative to data_args.video_folder first
        image_root_abs = os.path.join(data_args.video_folder or ".", spar_image_root_rel)
        # If not found, try relative to the manifest file's directory
        if not os.path.exists(image_root_abs):  # Check existence of the *directory*
            manifest_dir = os.path.dirname(manifest_path)
            image_root_abs_alt = os.path.join(manifest_dir, spar_image_root_rel)
            # Use the alternative path if it exists as a directory
            if os.path.isdir(image_root_abs_alt):
                image_root_abs = image_root_abs_alt
            else:  # Fallback to original path even if non-existent for consistency
                image_root_abs = os.path.join(data_args.video_folder or ".", spar_image_root_rel)

    else:
        image_root_abs = spar_image_root_rel

    if not os.path.exists(annotation_path):
        rank0_print(
            f" Warning: Annotation file not found: {annotation_path}. Searched relative to video_folder/current_dir and manifest dir. Skipping SPAR task {task_key}.")
        return []

    rank0_print(
        f"  Loading SPAR task: {task_key} | Annot: {annotation_path} | Root: {image_root_abs} | Repeat: {repeat_time:.2f}")
    task_data_raw = []
    try:
        with open(annotation_path, "r", encoding='utf-8') as f_jsonl:
            for i, line in enumerate(f_jsonl):
                try:
                    data_item = json.loads(line.strip())
                    task_data_raw.append(data_item)
                except json.JSONDecodeError:
                    rank0_print(f"  Warning: Skipping invalid JSON line {i + 1} in {annotation_path}")
    except Exception as e:
        rank0_print(f"  Error reading annotation file {annotation_path}: {e}")
        return []  # Skip this task if file reading fails

    # Apply repeat_time logic
    processed_task_data_raw = []
    original_task_len = len(task_data_raw)
    if original_task_len > 0:
        target_len = math.ceil(original_task_len * repeat_time)
        if repeat_time < 1.0 and repeat_time > 0.0:
            num_samples = target_len
            indices = random.sample(range(original_task_len), num_samples)
            processed_task_data_raw = [task_data_raw[i] for i in indices]
            # rank0_print(f"  Sampled {len(processed_task_data_raw)} items (ratio {repeat_time:.2f})")
        elif repeat_time > 1.0:
            repetitions = math.ceil(repeat_time)
            processed_task_data_raw = (task_data_raw * repetitions)[:target_len]
            # rank0_print(f"  Repeated to {len(processed_task_data_raw)} items (ratio {repeat_time:.2f})")
        elif repeat_time == 1.0:
            processed_task_data_raw = task_data_raw
            # rank0_print(f"  Using all {len(processed_task_data_raw)} items (ratio 1.0)")
        else:  # repeat_time <= 0 or invalid
            rank0_print(f"  Warning: Invalid repeat_time ({repeat_time:.2f}) for task {task_key}. Skipping task data.")
            processed_task_data_raw = []

    # Add metadata AFTER sampling/repetition
    task_data_processed = []
    for item in processed_task_data_raw:
        item['_is_spar'] = True
        item['_image_root'] = image_root_abs  # Use potentially resolved absolute path
        item['_annotation_path'] = annotation_path  # Source annotation file
        if 'type' not in item: item['type'] = task_type  # Add task type if missing
        task_data_processed.append(item)

    rank0_print(
        f"  -> Processed {len(task_data_processed)} samples for SPAR task {task_key} (Original: {original_task_len}, Repeat: {repeat_time:.2f})")
    return task_data_processed


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={
        "help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={
            "help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer
    pretrain_fusion_block: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pretrained .bin file for the fusion_block (e.g., SVPIFusionBlock)."}
    )
    ## spatial encoder
    spatial_tower: Optional[str] = field(default=None)
    spatial_tower_select_feature: Optional[str] = field(
        default=None, metadata={"help": 'Could be "last_hidden_state", "last_hidden_state,camera_tokens"'}
    )
    spatial_tower_select_layer: Optional[int] = field(default=-1)
    spatial_feature_dim: Optional[int] = field(default=None)
    tune_spatial_tower: bool = field(default=False)
    ## fusion block
    fusion_block: Optional[str] = field(default=None)
    tune_fusion_block: bool = field(default=False)
    fusion_block_pretrained: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HF repo id for the pretrained Mamba (used by MIDI fusion block)."})

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)

    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
        "help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)
    context_n_tor: Optional[int] = field(default=0)
    training_stage: Optional[str] = field(default=None)
    use_3dbox: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True,
                               metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",
                            metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    # fusion block lr
    fusion_block_lr: Optional[float] = None

    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    find_unused_parameters=True
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2",
                                     metadata={"help": "Use transformers attention implementation."})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'spatial_tower', 'fusion_block']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):

#     """Collects the state dict and dump to disk."""
#     if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
#         check_only_save_mm_adapter_tunnable = True
#     if hasattr(trainer.args, "tune_fusion_block") and trainer.args.tune_fusion_block:
#         check_only_save_mm_adapter_tunnable = True
#     # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
#     elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and (
#         "mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
#         check_only_save_mm_adapter_tunnable = True
#     else:
#         check_only_save_mm_adapter_tunnable = False

#     trainer.accelerator.wait_for_everyone()
#     torch.cuda.synchronize()
#     rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
#     if check_only_save_mm_adapter_tunnable:
#         # Only save Adapter
#         keys_to_match = ["mm_projector", "vision_resampler", "fusion_block"]  # save fusion_block and projectors
#         if getattr(trainer.args, "use_im_start_end", False):
#             keys_to_match.extend(["embed_tokens", "embed_in"])

#         weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
#         trainer.model.config.save_pretrained(output_dir)

#         current_folder = output_dir.split("/")[-1]
#         parent_folder = os.path.dirname(output_dir)
#         if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
#             if current_folder.startswith("checkpoint-"):
#                 mm_projector_folder = os.path.join(parent_folder, "mm_projector")
#                 os.makedirs(mm_projector_folder, exist_ok=True)
#                 torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
#             else:
#                 torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
#         return

#     # if trainer.deepspeed:
#     #     trainer.save_model(output_dir)
#     #     return

#     state_dict = trainer.model.state_dict()
#     # import pdb; pdb.set_trace()
#     keywords_to_exclude = ['spatial_tower']
#     filtered_state_dict = {}
#     for key, value in state_dict.items():
#         if not any(keyword in key for keyword in keywords_to_exclude):
#             filtered_state_dict[key] = value
#     state_dict = filtered_state_dict
#     rank0_print("Filtered out spatial_tower from the checkpoint.")
#     if trainer.args.should_save:
#         cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    (修正版 2 - 精确匹配)
    智能保存模型：
    - 如果 mm_tunable_parts == "fusion_block"，只保存 "fusion_block.bin"。
    - 如果 mm_tunable_parts == "mm_mlp_adapter"，只保存 "mm_projector.bin"。
    - 否则，保存完整的模型（但过滤掉 'spatial_tower'）。
    """

    check_only_save_adapter = False
    keys_to_match = []
    saved_parts_name = "full_model.bin" # 默认

    # --- [核心修正：根据您的输入进行精确匹配] ---
    
    # 1. 获取训练参数字符串
    tunable_parts_str = getattr(trainer.args, "mm_tunable_parts", None)
    
    if tunable_parts_str == "fusion_block":
        # 情况 1: 您正在训练您的 SVPI 模型
        rank0_print("Detected tunable part: 'fusion_block'. Setting to adapter-only save.")
        check_only_save_adapter = True
        keys_to_match = ["fusion_block"]
        saved_parts_name = "fusion_block.bin"
        
    elif tunable_parts_str == "mm_mlp_adapter":
        # 情况 2: 您正在训练基线 (Baseline)
        rank0_print("Detected tunable part: 'mm_mlp_adapter'. Setting to adapter-only save.")
        check_only_save_adapter = True
        keys_to_match = ["mm_projector"] # mm_mlp_adapter 对应于模型中的 mm_projector
        saved_parts_name = "mm_projector.bin"
        
    elif tunable_parts_str == "mm_vision_resampler":
        # (为未来保留，以防您需要)
        rank0_print("Detected tunable part: 'mm_vision_resampler'. Setting to adapter-only save.")
        check_only_save_adapter = True
        keys_to_match = ["vision_resampler"]
        saved_parts_name = "vision_resampler.bin"
    
    # (如果 tunable_parts_str 是 "llm" 或是 None, 
    #  check_only_save_adapter 将保持 False, 触发完整模型保存)
        
    # --- [修正结束] ---

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Adapter-only save mode detected: {check_only_save_adapter}")

    if check_only_save_adapter:
        # 确保我们真的找到了要保存的东西
        if not keys_to_match:
             rank0_print("Warning: Adapter-only mode set, but no matching keys found. Defaulting to full save.")
             check_only_save_adapter = False # 退回
        else:
            rank0_print(f"Adapter-only save mode. Saving keys matching: {keys_to_match}")
            weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
            
            # 始终保存 config
            trainer.model.config.save_pretrained(output_dir)

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                # 使用我们上面确定的 saved_parts_name
                if current_folder.startswith("checkpoint-"):
                    # 保存 checkpoint
                    connector_folder = os.path.join(parent_folder, "mm_connectors_bin")
                    os.makedirs(connector_folder, exist_ok=True)
                    save_path = os.path.join(connector_folder, f"{current_folder}_{saved_parts_name}")
                else:
                    # 保存最终模型
                    save_path = os.path.join(output_dir, saved_parts_name)
                
                rank0_print(f"Saving adapter weights to {save_path}")
                torch.save(weight_to_save, save_path)
            
            return # <-- 关键：保存完适配器后立即返回

    # --- 完整的模型保存逻辑 (如果不是 Adapter-only) ---
    rank0_print("Full model save mode detected.")
    
    # (您原来的完整保存逻辑，过滤 'spatial_tower'，这是正确的)
    state_dict = trainer.model.state_dict()
    keywords_to_exclude = ['spatial_tower'] # 始终过滤空间塔
    filtered_state_dict = {}
    keys_excluded = 0
    for key, value in state_dict.items():
        if not any(keyword in key for keyword in keywords_to_exclude):
            filtered_state_dict[key] = value
        else:
            keys_excluded += 1
            
    if keys_excluded > 0:
        rank0_print(f"Filtered out {keys_excluded} keys containing: {keywords_to_exclude}")
        
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in filtered_state_dict.items()}
        del state_dict
        del filtered_state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in
                                    tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2: cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(
                DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN,
                                                                  "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer,
                     has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx: conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful assistant.",
    data_args: DataArguments = None,
) -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}
    #  print("====preprocess_qwen======")
    # 用 deepcopy，避免修改外部 tokenizer
    tokenizer = copy.deepcopy(tokenizer)

    n_tor = data_args.context_n_tor if data_args is not None else None
    training_stage = data_args.training_stage if data_args is not None else None

    # 当有图像时，确保图像占位符被注册为 special token
    if has_image:
        try:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
        except Exception:
            pass
    if n_tor:
        try:
            tokenizer.add_tokens(["<tor>"], special_tokens=True)
        except Exception:
            pass
    try:
        tokenizer.add_tokens(["<3DBOX>"], special_tokens=True)
    except Exception:
        pass

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    tor_token_index = tokenizer.convert_tokens_to_ids("<tor>")
    bbox_3d_token_index = tokenizer.convert_tokens_to_ids("<3DBOX>")

    # 健壮获取 im_start / im_end
    add_tokens = getattr(tokenizer, "additional_special_tokens", []) or []
    add_ids = getattr(tokenizer, "additional_special_tokens_ids", []) or []
    token_to_id = {tok: tid for tok, tid in zip(add_tokens, add_ids)}

    im_start_token = "<|im_start|>"
    im_end_token = "<|im_end|>"

    im_start = token_to_id.get(im_start_token, None)
    im_end = token_to_id.get(im_end_token, None)

    # 兜底：若没找到命名 token，但 ids 至少有两个，则取前两个；否则禁用起止对
    if (im_start is None or im_end is None) and isinstance(add_ids, (list, tuple)) and len(add_ids) >= 2:
        # 注意：如果列表里包含了 <image> 的 id，此处仅作为兜底；真正的模板还是基于字符串 '<|im_start|>'/'<|im_end|>'
        im_start, im_end = add_ids[0], add_ids[1]
    if im_start is None or im_end is None:
        # 禁用起止对，仅保留换行 198 作为“反掩码”
        unmask_tokens_idx = [198]
    else:
        # 198 是换行；把起止对 id 放入不掩码集合
        unmask_tokens_idx = [198, im_start, im_end]

    nl_tokens = tokenizer("\n").input_ids

    # 重置 Qwen chat template（不每次都加 system）
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # system
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # 兼容 llava 数据键名
            try:
                role = conv["role"]
                content = conv["content"]
            except Exception:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
        
            if role == "assistant" and training_stage == "reasoning":
                if n_tor is None:
                    raise ValueError("n_tor must be set in reasoning stage.")
                content = insert_tor(content, n_tor)  # 添加tor标记

            # 添加嵌入逻辑
            elif role == "user" and training_stage == "vlm":
                if n_tor is not None and n_tor > 0:
                    # print("add <tor> to qwen llm!")
                    # 添加一个空格确保分词正确
                    tor_string = "<tor>" * n_tor 
                    content = content + tor_string

            enc = tokenizer.apply_chat_template([{"role": role, "content": content}])
            input_id += enc
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(enc)
            else:
                target += enc

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        # 反掩码特殊 token，并把 <image> 替换为 IMAGE_TOKEN_INDEX
        for idx, tid in enumerate(input_id):
            if tid in unmask_tokens_idx:
                target[idx] = tid
            if tid == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if n_tor and tid == tor_token_index:
                input_id[idx] = TOR_TOKEN_INDEX
                target[idx] = IGNORE_INDEX
            if tid == bbox_3d_token_index:
                input_id[idx] = BOX3D_TOKEN_INDEX
                target[idx] = IGNORE_INDEX
        input_ids.append(input_id)
        targets.append(target)
   # print(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(input_ids=input_ids, labels=targets)



def preprocess_mamba(
    sources: List[List[Dict]], 
    tokenizer: transformers.PreTrainedTokenizer, 
    max_len: int = 2048,
    system_message: str = "You are a helpful assistant.",
    data_args: Optional[DataArguments] = None, # 传入 data_args 来获取 n_tor 和 stage
) -> Dict:
    """
    这个版本的 preprocess_mamba 模仿 preprocess_qwen 的 *完整对话* 处理逻辑，
    但 *实现* 了 reasoning/vlm 两阶段的 <tor> 插入和标签逻辑。
    """
    
    # 1. 从 data_args 获取参数 (同 preprocess_qwen)
    roles = {"human": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)

    n_tor = data_args.context_n_tor if data_args is not None else None
    training_stage = data_args.training_stage if data_args is not None else None

    # 2. 为 Mamba 分词器添加 <tor> (同 preprocess_qwen)
    tor_token_index = -1
    if n_tor and n_tor > 0:
        try:
            tokenizer.add_tokens(["<tor>"], special_tokens=True)
            tor_token_index = tokenizer.convert_tokens_to_ids("<tor>")
            if tor_token_index == tokenizer.unk_token_id:
               print("警告: Mamba tokenizer 无法识别 <tor> token。")
               n_tor = 0 # 禁用 tor 功能
        except Exception:
             n_tor = 0
    bbox_3d_token_index = -1
    try:
        tokenizer.add_tokens(["<3DBOX>"], special_tokens=True)
        bbox_3d_token_index = tokenizer.convert_tokens_to_ids("<3DBOX>")
        if bbox_3d_token_index == tokenizer.unk_token_id:
             print("警告: Mamba tokenizer 无法识别 <3DBOX> token。")
             bbox_3d_token_index = -1
    except Exception:
        pass
    

    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for source in sources: # source 现在是 [ {turn1}, {turn2}, ... ]
        try:
            if not source: # 检查空对话
                print("警告: Mamba source 为空，跳过。")
                continue
            first_role = source[0].get("from", source[0].get("role"))
            if roles.get(first_role) != roles["human"]:
                source = source[1:]
            if not source: # 如果切片后为空
                print("警告: Mamba source 切片后为空，跳过。")
                continue
        except Exception as e:
            print(f"警告: 检查 Mamba source 起始角色时出错: {e}，跳过。")
            continue

        input_id, target = [], []

        # 5. 添加 System 消息 (同 preprocess_qwen, 但无模板)
        system_text = f"{system_message}\n" 
        system_tok = tokenizer(system_text, add_special_tokens=False)
        
        input_id += system_tok["input_ids"]
        target += [IGNORE_INDEX] * len(system_tok["input_ids"])

        # 6. 循环处理 *所有* 对话轮次 (同 preprocess_qwen)
        for conv in source:
            try:
                role = conv.get("role", conv.get("from"))
                content = conv.get("content", conv.get("value"))
            except Exception:
                continue 
            
            if role is None or content is None:
                continue

            role = roles.get(role, role)

            # 7. 准备文本内容 (根据 stage 修改)
            
            if role == "user":
                content = content.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            
            elif role == "assistant":
                # *** 修改点 1: 只在 reasoning 阶段添加 <tor> ***
                if training_stage == "reasoning":
                    if n_tor is None:
                        raise ValueError("n_tor must be set in reasoning stage.")
                    content = insert_tor(content, n_tor)
                # 在 "vlm" 阶段，保持 assistant 内容不变

            content_text = f"{content}\n"
            
            # 8. Tokenize 并设置标签
            enc_ids = tokenizer(content_text, add_special_tokens=False)["input_ids"]
            
            if not enc_ids: # 跳过空内容
                continue
                
            input_id += enc_ids
            
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(enc_ids)
            elif role == "assistant":
                if training_stage == "reasoning":
                    # 'reasoning' 阶段：预测 assistant (稍后会忽略 <tor>)
                    target += enc_ids
                else: # 'vlm' 阶段
                    # 'vlm' 阶段：不预测 assistant
                    target += [IGNORE_INDEX] * len(enc_ids)
            
            
            if role == "user" and training_stage == "vlm" and n_tor and n_tor > 0:
                # 这完全符合您的要求：“给提问加一个空字符串，然后应用insert_tor”
                tor_content = insert_tor("", n_tor)
                tor_content_text = f"{tor_content}\n"
                
                tor_enc_ids = tokenizer(tor_content_text, add_special_tokens=False)["input_ids"]
                if tor_enc_ids:
                    input_id += tor_enc_ids
                    # <tor> 标记本身从不被预测
                    target += [IGNORE_INDEX] * len(tor_enc_ids)
            # *** 结束修改点 3 ***


        if len(input_id) != len(target):
            print(f"Mamba-full-chat 长度不匹配: {len(input_id)} != {len(target)}。跳过此样本。")
            continue

        # 9. 后处理：*仅在 reasoning 阶段* 忽略 <tor> 标签
        # (vlm 阶段的 <tor> 已经被设为 IGNORE_INDEX)
        if training_stage == "reasoning" and n_tor and n_tor > 0 and tor_token_index != -1:
            for idx, tid in enumerate(input_id):
                if tid == tor_token_index:
                    target[idx] = IGNORE_INDEX
        if bbox_3d_token_index != -1:
            for idx, tid in enumerate(input_id):
                if tid == bbox_3d_token_index:
                    target[idx] = BOX3D_TOKEN_INDEX

        # 10. 截断
        if len(input_id) > max_len:
            input_id = input_id[:max_len]
            target = target[:max_len]
            
        input_ids_list.append(input_id)
        labels_list.append(target)

    # --- 11. 批处理和填充 ---
    if not input_ids_list:
        print("警告: preprocess_mamba 未生成任何有效输入。")
        pad_id = tokenizer.pad_token_id or 0
        return dict(input_ids=torch.tensor([[pad_id]], dtype=torch.long), 
                    labels=torch.tensor([[IGNORE_INDEX]], dtype=torch.long))

    max_batch_len = max(len(ids) for ids in input_ids_list)
    
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
            
    padded_input_ids = []
    padded_labels = []

    for ids, labs in zip(input_ids_list, labels_list):
        pad_count = max_batch_len - len(ids)
        padded_input_ids.append(ids + [pad_token_id] * pad_count)
        padded_labels.append(labs + [IGNORE_INDEX] * pad_count)

    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(padded_labels, dtype=torch.long)

    return dict(input_ids=input_ids_tensor, labels=labels_tensor)


def preprocess(sources: Sequence[str],
               tokenizer: transformers.PreTrainedTokenizer,
               has_image: bool = False,
               data_args: DataArguments = None) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image, data_args=data_args)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments,
                 mamba_tokenizer: Optional[transformers.PreTrainedTokenizer] = None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.mamba_tokenizer = mamba_tokenizer
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                loaded_count = 0  # 用于统计总加载量
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")

                    is_spar = dataset.get("is_spar", False)  # <--- 获取 is_spar 标志
                    with_depth = dataset.get("with_depth", False)
                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy. Is SPAR: {is_spar}")
                    

                    current_batch_data = []  # 存储当前文件/清单加载的数据

                   
                    try:
                        if is_spar:
                            # --- SPAR Manifest Loading (Parallelized) ---
                            rank0_print(f" Processing SPAR manifest: {json_path}")
                            with open(json_path, "r") as f_manifest:
                                spar_manifest_content = json.load(f_manifest)

                            processed_tasks_data = []
                            # Adjust num_workers as needed. os.cpu_count() might be too high.
                            num_workers = min(16, os.cpu_count() or 1)
                            rank0_print(f" Using {num_workers} workers for SPAR task processing.")
                            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                                # Submit all tasks
                                future_to_task = {
                                    executor.submit(_process_spar_task, task_key, task_meta, json_path,
                                                    data_args): task_key
                                    for task_key, task_meta in spar_manifest_content.items()
                                }

                                # Collect results as they complete
                                for future in concurrent.futures.as_completed(future_to_task):
                                    task_key = future_to_task[future]
                                    try:
                                        task_result = future.result()  # Get result from future
                                        if task_result:  # Avoid extending with empty lists if task failed/skipped
                                            processed_tasks_data.extend(task_result)
                                    except Exception as exc:
                                        rank0_print(
                                            f' SPAR task {task_key} generated an exception during execution: {exc}')
                                        # Optionally, re-raise, log more details, or implement specific error handling

                            current_batch_data.extend(processed_tasks_data)
                            rank0_print(
                                f" Finished processing SPAR manifest {json_path}, gathered {len(processed_tasks_data)} items total.")

                        else:
                            # --- 原有的非 SPAR 数据加载逻辑 (.jsonl / .json) ---
                            cur_data_dict_non_spar = []
                            if json_path.endswith(".jsonl"):
                                with open(json_path, "r") as json_file:
                                    for line in json_file:
                                        try:
                                            cur_data_dict_non_spar.append(json.loads(line.strip()))
                                        except json.JSONDecodeError:
                                            pass
                            elif json_path.endswith(".json"):
                                with open(json_path, "r") as json_file:
                                    cur_data_dict_non_spar = json.load(json_file)
                            else:
                                raise ValueError(f"Unsupported file type for non-SPAR data: {json_path}")

                            # 添加元数据
                            for item in cur_data_dict_non_spar:
                                item['_is_spar'] = False
                                item['_with_depth'] = with_depth
                                item['_annotation_path'] = json_path
                            current_batch_data = cur_data_dict_non_spar  # 存储加载的数据

                    except Exception as e:
                        rank0_print(f"Error processing file/manifest {json_path}: {e}")
                        continue  # 跳过这个出错的文件/清单

                    media_type = dataset.get("media_type", None)
                    if media_type is None:
                        if 'image' in current_batch_data[0].keys(): # NOTE 碰到混合数据可能会出错
                            media_type = 'image'
                        elif 'video' in current_batch_data[0].keys():
                            media_type = 'video'
                        else:
                            media_type = 'text'

                    # --- 对非 SPAR 数据应用采样策略 ---
                    # (SPAR 数据的采样通常由其内部 repeat_time 控制，这里不对其应用 YAML 采样)
                    if not is_spar and len(current_batch_data) > 0:
                        original_len = len(current_batch_data)
                        sampling_number = None
                        if ":" in sampling_strategy:
                            sampling_strategy_part, sampling_number_str = sampling_strategy.split(":")
                            try:
                                if "%" in sampling_number_str:
                                    percentage = int(sampling_number_str.strip('%'))
                                    sampling_number = math.ceil(percentage * original_len / 100)
                                else:
                                    sampling_number = int(sampling_number_str)

                                if sampling_strategy_part == "first" and sampling_number is not None:
                                    current_batch_data = current_batch_data[:sampling_number]
                                elif sampling_strategy_part == "end" and sampling_number is not None:
                                    current_batch_data = current_batch_data[-sampling_number:]
                                elif sampling_strategy_part == "random" and sampling_number is not None:
                                    random.shuffle(current_batch_data)
                                    current_batch_data = current_batch_data[:sampling_number]
                            except ValueError:
                                rank0_print(
                                    f"Warning: Invalid sampling number format in '{sampling_strategy}'. Loading all data.")
                            

                        rank0_print(
                            f"Loaded {len(current_batch_data)} samples from {json_path} (original: {original_len}) after sampling.")


                    data_root = dataset.get("data_root", "")
                    if data_root != "":
                        rank0_print(f" Prepending data_root '{data_root}' to media paths for media_type '{media_type}'.")
                    for i in range(len(current_batch_data)):
                        current_batch_data[i][media_type] = os.path.join(data_root, current_batch_data[i][media_type])

                    # 将当前批次加载并处理（采样）后的数据添加到总列表
                    self.list_data_dict.extend(current_batch_data)
                    loaded_count += len(current_batch_data)  # 更新总计数
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)
        # shuffle the data
        random.shuffle(self.list_data_dict)
        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        if self.mamba_tokenizer is not None:
            try:
                tor_id = self.mamba_tokenizer.convert_tokens_to_ids("<tor>")
                if tor_id is None or tor_id == self.mamba_tokenizer.unk_token_id:
                    rank0_print("Warning: Mamba tokenizer did not have <tor> registered. Adding it.")
                    self.mamba_tokenizer.add_tokens(["<tor>"], special_tokens=True)
                    tor_id = self.mamba_tokenizer.convert_tokens_to_ids("<tor>")

                # 将 mamba_tor_id 存储为类成员
                self.mamba_tor_id = tor_id
                rank0_print(f"LazySupervisedDataset: Mamba <tor> ID set to: {self.mamba_tor_id}")
            except Exception as e:
                rank0_print(f"Error registering <tor> in Mamba tokenizer: {e}")
                raise e

        self.data_folder_map = {'scannet': 'ScanNet', 'scannetpp': 'ScanNetpp', 'arkit': 'ARkitScenes'}
        self.data_intrinsics_map = {'scannet': 'intrinsic_depth_', 'scannetpp': 'intrinsics_',
                                    'arkit': 'arkit_intrinsics_'}
        self.video_fps = {'scannet': 30, 'scannetpp': 60, 'arkit': 30}

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        # import pdb; pdb.set_trace()
        n_tor = self.data_args.context_n_tor
        data_item = self.list_data_dict[i]
        use_3dbox = self.data_args.use_3dbox if hasattr(self.data_args, 'use_3dbox') else False
        gt_numerical_answers = []
        if use_3dbox:
            if "conversations" in data_item:
                for conv in data_item["conversations"]:
                    role = conv.get("from", conv.get("role"))
                    if role == "gpt" or role == "assistant":
                        content = conv.get("value", conv.get("content"))
                        # 尝试对数值答案进行标记
                        try:
                            float_val = float(content.strip())
                            conv["value"] = "<3DBOX>" 
                            gt_numerical_answers.append(float_val)
                        except (ValueError, TypeError):
                            pass
        else:
            gt_numerical_answers = []                    

        if data_item is not list:
            sources = [data_item]
        else:
            sources = data_item

        processor = self.data_args.image_processor  # Get processor from data_args
        image = []
        point_maps = []
        # --- Image Processing Branch ---
        if "image" in data_item:
            image_files = data_item["image"]
            if not isinstance(image_files, list):
                image_files = [image_files]

            # # Determine correct image folder/root
            # if data_item.get('_is_spar', False) and data_item.get('_image_root'):
            #     image_folder = data_item['_image_root']
            # else:
            #     # Fallback if not SPAR or _image_root is missing
            #     image_folder = self.data_args.image_folder

            # 1. Load all PIL images first
            pil_images = []
            relative_image_paths = []  # Store relative paths for potential use in drawing logic
            for image_file in image_files:
                relative_image_paths.append(image_file)
                try:
                    # Construct the full path
                    full_image_path = image_file
                    # If image_file is not absolute, join with the determined image_folder
                    if not os.path.isabs(image_file):
                        # 有dataroot下，这个应该不执行
                        if not image_folder:
                            rank0_print(
                                f"Warning: Image folder is not set for relative path '{image_file}' in item {i}. Trying current dir.")
                            # Decide fallback: use current dir, or skip?
                            # full_image_path = os.path.join(".", image_file) # Example: current dir
                            raise ValueError(f"Image folder not set for relative path: {image_file}")
                        else:
                            full_image_path = os.path.join(image_folder, image_file)

                    # Check existence and load
                    if not os.path.exists(full_image_path):
                        # Optional: Try path relative to annotation file for SPAR
                        alt_path_checked = False
                        if '_annotation_path' in data_item and not os.path.isabs(image_file):
                            base_dir = os.path.dirname(data_item['_annotation_path'])
                            alt_path = os.path.join(base_dir, image_file)  # Assume image path is relative to annotation
                            if os.path.exists(alt_path):
                                full_image_path = alt_path
                                alt_path_checked = True

                        if not os.path.exists(full_image_path) and not alt_path_checked:
                            raise FileNotFoundError(
                                f"Image file not found at '{full_image_path}' or alternative relative path for item {i}")

                    # print(f"Loading image: {full_image_path}") # Debug
                    img = Image.open(full_image_path).convert("RGB")
                    pil_images.append(img)

                except Exception as e:
                    rank0_print(
                        f"ERROR loading image '{image_file}' at path '{full_image_path}' for item {i}. Error: {e}")
                    # Decide how to handle: raise error, return dummy item, or append None and skip processing this image?
                    pil_images.append(None)  # Append None as a placeholder

            # Filter out None entries if any image failed loading
            valid_pil_images = [(img, path) for img, path in zip(pil_images, relative_image_paths) if img is not None]
            if not valid_pil_images:
                raise RuntimeError(f"Could not load ANY valid images for item {i}. Files: {image_files}")

            # 2. Apply Marker Drawing (if SPAR and applicable)
            # Pass the list of *valid* PIL images to the drawing function
            current_pil_list = [img for img, path in valid_pil_images]  # Get just the images for drawing
            if HAS_DRAW_MARKER and data_item.get('_is_spar', False):
                task_type = data_item.get('type', None)
                draw_fn = DRAW_FUNCTIONS.get(task_type)
                if draw_fn:
                    # rank0_print(f"Applying draw function '{task_type}' for item {i}")
                    try:
                        # Draw function modifies images in current_pil_list in-place
                        if len(current_pil_list) == 1:
                            draw_fn(current_pil_list[0], data_item)
                        else:
                            draw_fn(current_pil_list, data_item)
                    except Exception as e:
                        rank0_print(f"ERROR applying draw function {task_type} for item {i}: {e}")
                        # Decide whether to continue with potentially un-drawn images or raise error

            # 3. Process each PIL image (original or drawn) using logic similar to original process_image
            for idx, (pil_img, rel_path) in enumerate(valid_pil_images):
                # pil_img is now potentially modified by draw_fn if it was SPAR data

                image_size = pil_img.size
                # Determine aspect ratio override (e.g., 'pad' for multi-image)
                # Using len(image_files) > 1 as the condition for 'pad' override
                overwrite_aspect_ratio = "pad" if len(image_files) > 1 else None

                # Get aspect ratio setting from data_args, potentially overridden
                image_aspect_ratio = self.data_args.image_aspect_ratio
                if overwrite_aspect_ratio is not None:
                    image_aspect_ratio = overwrite_aspect_ratio

                # Apply aspect ratio processing and preprocessing
                processed_tensor = None
                try:
                    if image_aspect_ratio == "highres":
                        processed_tensor = process_highres_image(pil_img, processor,
                                                                 self.data_args.image_grid_pinpoints)
                    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        processed_tensor = process_anyres_image(pil_img, processor, self.data_args.image_grid_pinpoints)
                    elif image_aspect_ratio == "crop_split":
                        processed_tensor = process_highres_image_crop_split(pil_img, self.data_args)
                    elif image_aspect_ratio == "pad":
                        def expand2square(img, background_color):
                            width, height = img.size
                            if width == height:
                                return img
                            elif width > height:
                                result = Image.new(img.mode, (width, width), background_color)
                                result.paste(img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(img.mode, (height, height), background_color)
                                result.paste(img, ((height - width) // 2, 0))
                                return result

                        background = tuple(
                            int(x * 255) for x in getattr(processor, 'image_mean', [0.485, 0.456, 0.406]))
                        image_padded = expand2square(pil_img, background)
                        processed_tensor = processor.preprocess(image_padded, return_tensors="pt")["pixel_values"][0]
                    else:  # Default behavior (simple resize/preprocess)
                        processed_tensor = processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]

                    image.append((processed_tensor, image_size, "image"))

                except Exception as e:
                    rank0_print(
                        f"ERROR during preprocessing image {idx} ({rel_path}) for item {i} with aspect ratio '{image_aspect_ratio}'. Error: {e}")
                    # Append a dummy tensor or handle error
                    crop_size = self.data_args.image_processor.crop_size
                    dummy_tensor = torch.zeros(1, 3, crop_size["height"], crop_size["width"])
                    image.append((dummy_tensor, image_size, "image_preprocess_failed"))
            if len(image) > 8:
                try:
                    video_like_tensor = torch.stack([img[0] for img in image], dim=0)  # Stack to (N, C, H, W)
                    # Use first image size as representative
                    rep_size = image[0][1]
                    image = [(video_like_tensor, rep_size, "video")]  # Length 1 list
                    num_image_tokens = 1  # Video convention: 1 token
                except Exception as stack_e:
                    rank0_print(f"Error stacking image tensors for item {i}: {stack_e}")
            else:
                num_image_tokens = len(image)
            # Process text conversations after handling images
            # Pass data_args which might contain multimodal processing settings
            # change the token num of conversations
            image_tokens_str = (DEFAULT_IMAGE_TOKEN + "\n") * num_image_tokens
            cleaned_original_text = data_item["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            data_item["conversations"][0]["value"] = f'{image_tokens_str}{cleaned_original_text}'
            sources = preprocess_multimodal(copy.deepcopy([data_item["conversations"]]), self.data_args)

        elif "video" in sources[0] and not data_item.get('_with_depth', False):
            video_file = self.list_data_dict[i]["video"]
            # video_folder = self.data_args.video_folder
            # video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if
                                   os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2

                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

                    frame_time = [i / 2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                else:
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file,
                                                                                                    self.data_args)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0][
                        "value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)

        elif "video" in sources[0] and data_item.get('_with_depth', False):
            raise(NotImplementedError("Video with depth processing is not implemented yet."))
        else:
            # --- Original Text-Only Logic ---
            # This branch handles cases where 'image', 'video' (non-depth), are not present
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image, data_args=self.data_args)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])


        if n_tor is not None and n_tor > 0:
            original_conversations = data_item["conversations"]
            if self.data_args.training_stage is None:
                raise ValueError("data_args.training_stage must be specified when using Mamba preprocessing.")
            mamba_dict = preprocess_mamba(
                sources, self.mamba_tokenizer, data_args=self.data_args
            )
            data_dict["mamba_input_ids"] = mamba_dict["input_ids"][0]
            data_dict["mamba_labels"] = mamba_dict["labels"][0]
            data_dict["mamba_attention_mask"] = torch.tensor([1] * len(data_dict["mamba_input_ids"]),dtype=torch.long)

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]),
                 "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        # add spatial features
        # if "video" in self.list_data_dict[i]:
        #     video_folder = self.data_args.video_folder
        #     spatial_features_path = os.path.join(video_folder,
        #                                          self.list_data_dict[i]['video'].replace('.mp4', '.pt').replace(
        #                                              'videos', 'spatial_features'))
        #     if os.path.exists(spatial_features_path):
        #         spatial_features = torch.load(spatial_features_path)
        #         data_dict["spatial_features"] = spatial_features

        # add point cloud
        if "_with_depth" in self.list_data_dict[i] and self.list_data_dict[i]["_with_depth"]:
            data_dict["point_maps"] = point_maps

        if gt_numerical_answers:
            data_dict["gt_numerical_answers"] = torch.tensor(gt_numerical_answers, dtype=torch.float)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, mamba_tokenizer: Optional[transformers.PreTrainedTokenizer] = None):
        self.tokenizer = tokenizer
        self.mamba_tokenizer = mamba_tokenizer  

    def pad_sequence(self, input_ids, batch_first, padding_value, tokenizer):
        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, tokenizer=self.tokenizer)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX, tokenizer=self.tokenizer)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

    
        if "mamba_input_ids" in instances[0]:
            assert self.mamba_tokenizer is not None, "mamba_tokenizer must be provided in DataCollator if mamba_input_ids are present in the data."
            # 3a. 提取 Mamba 数据
            mamba_input_ids, mamba_labels, mamba_attention_mask = tuple(
                [instance[key] for instance in instances] for key in
                ("mamba_input_ids", "mamba_labels", "mamba_attention_mask"))

            # 3b. 截断 Mamba 数据
            # (注意: 如果 mamba_tokenizer 没有设置 model_max_length, 你可能需要硬编码一个)
            mamba_max_len = getattr(self.mamba_tokenizer, 'model_max_length', 2048)
            mamba_input_ids = [_ids[:mamba_max_len] for _ids in mamba_input_ids]
            mamba_labels = [_lbls[:mamba_max_len] for _lbls in mamba_labels]
            mamba_attention_mask = [_mask[:mamba_max_len] for _mask in mamba_attention_mask]

            # 获取 Mamba 填充值
            if self.mamba_tokenizer.pad_token_id is None:
                self.mamba_tokenizer.pad_token_id = 0  # (或 self.mamba_tokenizer.eos_token_id)

            mamba_input_pad_id = self.mamba_tokenizer.pad_token_id
            mamba_label_pad_id = IGNORE_INDEX
            mamba_mask_pad_id = 0  # Attention mask 用 0 填充

            mamba_input_ids = self.pad_sequence(
                mamba_input_ids, batch_first=True, padding_value=mamba_input_pad_id, tokenizer=self.mamba_tokenizer)
            mamba_labels = self.pad_sequence(
                mamba_labels, batch_first=True, padding_value=mamba_label_pad_id, tokenizer=self.mamba_tokenizer)
            mamba_attention_mask = self.pad_sequence(
                mamba_attention_mask, batch_first=True, padding_value=mamba_mask_pad_id, tokenizer=self.mamba_tokenizer)

            # 3e. 添加到批次中
            batch["mamba_input_ids"] = mamba_input_ids
            batch["mamba_labels"] = mamba_labels.long()
            batch["mamba_attention_mask"] = mamba_attention_mask

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
            # Image: (N, P, C, H, W)
            # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        # --> Add handling for spatial_features <--
        if "spatial_features" in instances[0]:
            spatial_features = [instance["spatial_features"] for instance in instances]
            # Assuming spatial_features are tensors that can be stacked
            # Adjust this logic if they need different handling (e.g., padding)
            if all(isinstance(sf, torch.Tensor) for sf in spatial_features):
                try:
                    batch['spatial_features'] = torch.stack(spatial_features)
                except Exception as e:
                    # Handle cases where stacking might fail (e.g., different shapes if not padded)
                    # You might need custom padding logic here depending on the feature structure
                    print(f"Warning: Could not stack spatial_features due to: {e}. Passing as list.")
                    batch['spatial_features'] = spatial_features
            else:
                # If features are not tensors or mixed types, pass as a list
                batch['spatial_features'] = spatial_features

        # add point maps
        if "point_maps" in instances[0]:
            point_maps = [instance["point_maps"] for instance in instances]
            batch["point_maps"] = point_maps

        if any("gt_numerical_answers" in instance for instance in instances):
            gt_nums_list = []
            for instance in instances:
                gt_val = instance.get("gt_numerical_answers") 
                if gt_val is not None:
                    gt_nums_list.append(gt_val)
            if gt_nums_list:
                max_len = max(t.size(0) for t in gt_nums_list)
                
                padded_gt_nums = torch.full(
                    (len(instances), max_len), 
                    IGNORE_INDEX, 
                    dtype=torch.float
                )
                
                for i, instance in enumerate(instances):
                    gt_val = instance.get("gt_numerical_answers")
                    if gt_val is not None:
                        padded_gt_nums[i, :gt_val.size(0)] = gt_val   
                batch["gt_numerical_answers"] = padded_gt_nums

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args,
                                mamba_tokenizer: transformers.PreTrainedTokenizer = None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, mamba_tokenizer=mamba_tokenizer,
                                          data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, mamba_tokenizer=mamba_tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args,data_args,  bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")
    # import pdb;pdb.set_trace()
    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.fusion_block_pretrained is not None:
        overwrite_config["fusion_block_pretrained"] = model_args.fusion_block_pretrained
    if data_args.training_stage is not None:
        overwrite_config["training_stage"] = data_args.training_stage

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(
            cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.spatial_tower is not None:
        overwrite_config["spatial_tower"] = model_args.spatial_tower
        overwrite_config["spatial_tower_select_feature"] = model_args.spatial_tower_select_feature
        overwrite_config["spatial_tower_select_layer"] = model_args.spatial_tower_select_layer
        overwrite_config["spatial_feature_dim"] = model_args.spatial_feature_dim

    if model_args.fusion_block is not None:
        overwrite_config["fusion_block"] = model_args.fusion_block

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    ignore_mismatched_sizes=True,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    ignore_mismatched_sizes=True,
                    **customized_kwargs,
                )
        elif "gemma" in model_args.model_name_or_path.lower():
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if getattr(training_args, "gradient_checkpointing", False):
        # 如果 TrainingArguments 里没有这个字段，也能 set 属性，不会报错
        gc_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None)
        if not isinstance(gc_kwargs, dict):
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
        # 强制覆盖/补齐 use_reentrant
            training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

    # 设置随机种子以保证可复现性
    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 为了更强的可复现性，可以考虑以下设置，但可能会牺牲性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = get_model(model_args, training_args, data_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            if isinstance(output, tuple):
                # 确保 output[0] 存在且是一个张量
                if output and isinstance(output[0], torch.Tensor):
                        output[0].requires_grad_(True)
                else:
                    output.requires_grad_(True)

            # [新的正确代码] 
            # 附加钩子到 LLM 的第一个 Transformer Layer
            try:
                # 假设 LLM 在 self.model 中，并且层在 .layers 属性
                # 这适用于 Qwen, Llama, Mistral 等
                first_block = model.get_model().layers[0] 
                first_block.register_forward_hook(make_inputs_require_grad)
                rank0_print("Successfully attached gradient_checkpointing hook to first layer.")
            except Exception as e:
                rank0_print(f"Failed to attach gradient_checkpointing hook: {e}")
                rank0_print("WARNING: Gradient Checkpointing may fail to propagate grads.")
            # model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                               cache_dir=training_args.cache_dir,
                                                               model_max_length=training_args.model_max_length,
                                                               padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                               cache_dir=training_args.cache_dir,
                                                               model_max_length=training_args.model_max_length,
                                                               padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.spatial_tower is not None:
        model.get_model().initialize_spatial_tower(model_args=model_args, fsdp=training_args.fsdp)
        spatial_tower = model.get_spatial_tower()
        spatial_tower.to(device=training_args.device)

        model.config.spatial_tower = model_args.spatial_tower
        model.config.spatial_tower_select_feature = model_args.spatial_tower_select_feature
        model.config.spatial_tower_select_layer = model_args.spatial_tower_select_layer
        model.config.spatial_feature_dim = model_args.spatial_feature_dim


    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in
                                  range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

    
    if model_args.fusion_block is not None:
        model.get_model().initialize_fusion_block(model_args=model_args, fsdp=training_args.fsdp)
        fusion_block = model.get_fusion_block()
        fusion_block.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.config.fusion_block = model_args.fusion_block
        model.config.fusion_block_lr = training_args.fusion_block_lr
        model.config.training_stage = data_args.training_stage

    
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position = model_args.mm_newline_position
    model.config.add_faster_video = model_args.add_faster_video
    model.config.faster_token_stride = model_args.faster_token_stride
    model.config.add_time_instruction = data_args.add_time_instruction
    model.config.force_sample = data_args.force_sample
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

    if data_args.context_n_tor is not None:
        model.config.context_n_tor = data_args.context_n_tor

    print("================================")
    print(model_args.mm_tunable_parts)
    ### Deciding train which part of the model
    if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
        model.config.tune_spatial_tower = training_args.tune_spatial_tower = model_args.tune_spatial_tower
        model.config.tune_fusion_block = training_args.tune_fusion_block = model_args.tune_fusion_block
        if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler or model_args.tune_fusion_block or model_args.tune_spatial_tower:
            model.requires_grad_(False)
        if training_args.lora_enable:
            for name, param in model.named_parameters():
                if 'lora_' in name:  # 不冻结LoRA参数
                    param.requires_grad = True
        if model_args.tune_spatial_tower:
            for p in model.get_spatial_tower().parameters():
                p.requires_grad = True
        if model_args.tune_fusion_block:
            for p in model.get_fusion_block().parameters():
                p.requires_grad = True
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if model_args.tune_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
        if training_args.freeze_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
        if model_args.unfreeze_mm_vision_tower:
            vision_tower.requires_grad_(True)
        else:
            vision_tower.requires_grad_(False)

    else:
        rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        # Set the entire model to not require gradients by default
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(False)
        model.get_model().vision_resampler.requires_grad_(False)

        if training_args.lora_enable:
            rank0_print("Re-unfreezing LoRA parameters...")
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True

        # Parse the mm_tunable_parts to decide which parts to unfreeze
        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_mlp_adapter" in tunable_parts:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if "mm_vision_resampler" in tunable_parts:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True
        if "mm_vision_tower" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad_(True)
        # if "mm_language_model" in tunable_parts:
        #     for name, param in model.named_parameters():
        #         if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
        #             param.requires_grad_(True)
        if "mm_language_model" in tunable_parts:
            print("tune language model")
        # 1. 定义所有非LLM模块的关键词
            non_llm_keywords = [
                'vision_tower', 
                'mm_projector', 
                'vision_resampler', 
                'spatial_tower', 
                'fusion_block'
            ]
            for name, param in model.named_parameters():
                if not any(keyword in name for keyword in non_llm_keywords):
                    if "q_proj" in name or "v_proj" in name:
                        param.requires_grad = True
                    if getattr(data_args, "use_3dbox", False):
                        if "numerical_answer_decoder" in name:
                            param.requires_grad = True
        if "spatial_tower" in tunable_parts:
            for p in model.get_spatial_tower().parameters():
                p.requires_grad = True
        if "fusion_block" in tunable_parts:
            for p in model.get_fusion_block().parameters():
                p.requires_grad = True


    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(
        p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    rank0_print(f"Total parameters: ~{total_params / 1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params / 1e6:.2f} MB)")
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if model_args.fusion_block_pretrained is not None:
        mamba_tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.fusion_block_pretrained,
                                                                     cache_dir=training_args.cache_dir,
                                                                     model_max_length=training_args.model_max_length,
                                                                     padding_side="right")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, mamba_tokenizer=(
        mamba_tokenizer if model_args.fusion_block_pretrained is not None else None))
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
