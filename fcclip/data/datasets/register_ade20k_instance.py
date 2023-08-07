"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_ade20k_instance.py
"""

import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

from . import openseg_classes
import copy
ADE_CATEGORIES = copy.deepcopy(openseg_classes.ADE20K_150_CATEGORIES)
ADE_CATEGORIES = [x for x in ADE_CATEGORIES if x["isthing"] == 1]

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "openvocab_ade20k_instance_train": (
        "ADEChallengeData2016/images/training",
        "ADEChallengeData2016/ade20k_instance_train.json",
    ),
    "openvocab_ade20k_instance_val": (
        "ADEChallengeData2016/images/validation",
        "ADEChallengeData2016/ade20k_instance_val.json",
    ),
}


def _get_ade_instances_meta():
    thing_ids = [k["id"] for k in ADE_CATEGORIES]
    assert len(thing_ids) == 100, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ADE_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_ade20k_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ade_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_instance(_root)
