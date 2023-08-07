"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_ade20k_full.py
"""

import os

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from . import openseg_classes

ADE20K_847_CATEGORIES = openseg_classes.get_ade20k_847_categories_with_prompt_eng()

ADE20k_847_COLORS = [np.random.randint(256, size=3).tolist() for k in ADE20K_847_CATEGORIES]

MetadataCatalog.get("openvocab_ade20k_full_sem_seg_train").set(
    stuff_colors=ADE20k_847_COLORS[:],
)

MetadataCatalog.get("openvocab_ade20k_full_sem_seg_val").set(
    stuff_colors=ADE20k_847_COLORS[:],
)


def _get_ade20k_847_meta():
    # We only need class names
    stuff_classes = [k["name"] for k in ADE20K_847_CATEGORIES]
    assert len(stuff_classes) == 847, len(stuff_classes)

    ret = {
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_ade20k_847(root):
    root = os.path.join(root, "ADE20K_2021_17_01")
    meta = _get_ade20k_847_meta()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images_detectron2", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"openvocab_ade20k_full_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
            gt_ext="tif",
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_847(_root)