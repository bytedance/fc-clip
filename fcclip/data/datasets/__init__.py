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

from . import (
    register_coco_panoptic_annos_semseg,
    register_ade20k_panoptic,
    register_cityscapes_panoptic,
    register_mapillary_vistas_panoptic,
    register_ade20k_full,
    register_pascal_voc_20_semantic,
    register_pascal_voc_21_semantic,
    register_pascal_ctx_59_sem_seg,
    register_pascal_ctx_459_sem_seg,
    register_coco_instance,
    register_ade20k_instance,
    register_coco_stuff_164k,
    openseg_classes
)
