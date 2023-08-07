"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import os

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from . import openseg_classes

PASCAL_CTX_459_CATEGORIES=openseg_classes.get_pascal_ctx_459_categories_with_prompt_eng()

PASCAL_CTX_459_COLORS = [k["color"] for k in PASCAL_CTX_459_CATEGORIES]

MetadataCatalog.get("openvocab_pascal_ctx459_sem_seg_train").set(
    stuff_colors=PASCAL_CTX_459_COLORS[:],
)

MetadataCatalog.get("openvocab_pascal_ctx459_sem_seg_val").set(
    stuff_colors=PASCAL_CTX_459_COLORS[:],
)

def _get_ctx459_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in PASCAL_CTX_459_CATEGORIES]
    assert len(stuff_ids) == 459, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in PASCAL_CTX_459_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_ctx459(root):
    root = os.path.join(root, "pascal_ctx_d2")
    meta = _get_ctx459_meta()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_ctx459", dirname)
        name = f"openvocab_pascal_ctx459_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            thing_dataset_id_to_contiguous_id={},  # to make Mask2Former happy
            stuff_dataset_id_to_contiguous_id=meta["stuff_dataset_id_to_contiguous_id"],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
            gt_ext="tif",
        )



        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ctx459(_root)