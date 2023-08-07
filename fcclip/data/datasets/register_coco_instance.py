"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/detectron2/blob/67ac149947124670f6678e1bdd75f89dbf0dd5e7/detectron2/data/datasets/coco.py
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
COCO_CATEGORIES = openseg_classes.get_coco_categories_with_prompt_eng()
COCO_CATEGORIES = [x for x in COCO_CATEGORIES if x["isthing"] == 1]

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "openvocab_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/instances_train2017.json",
    ),
    "openvocab_coco_2017_val": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
    ),
}


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_coco_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_coco_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_instance(_root)
