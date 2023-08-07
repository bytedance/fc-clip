"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/predictor.py
"""

import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch
import itertools


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor as d2_defaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
import detectron2.utils.visualizer as d2_visualizer


class DefaultPredictor(d2_defaultPredictor):

    def set_metadata(self, metadata):
        self.model.set_metadata(metadata)


class OpenVocabVisualizer(Visualizer):
    def draw_panoptic_seg(self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7):
        """
        Draw panoptic prediction annotations or results.

        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict] or None): Describe each segment in `panoptic_seg`.
                If it is a ``list[dict]``, each dict contains keys "id", "category_id".
                If None, category id of each pixel is computed by
                ``pixel // metadata.label_divisor``.
            area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

        Returns:
            output (VisImage): image object with visualizations.
        """
        pred = d2_visualizer._PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))
        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            text = self.metadata.stuff_classes[category_idx].split(',')[0]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=d2_visualizer._OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        # draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        stuff_classes = self.metadata.stuff_classes
        stuff_classes = [x.split(',')[0] for x in stuff_classes]
        labels = d2_visualizer._create_text_labels(
            category_ids, scores, stuff_classes, [x.get("iscrowd", 0) for x in sinfo]
        )

        try:
            colors = [
                self._jitter([x / 255 for x in self.metadata.stuff_colors[c]]) for c in category_ids
            ]
        except AttributeError:
            colors = None
        self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        return self.output


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        coco_metadata = MetadataCatalog.get("openvocab_coco_2017_val_panoptic_with_sem_seg")
        ade20k_metadata = MetadataCatalog.get("openvocab_ade20k_panoptic_val")
        lvis_classes = open("./fcclip/data/datasets/lvis_1203_with_prompt_eng.txt", 'r').read().splitlines()
        lvis_classes = [x[x.find(':')+1:] for x in lvis_classes]
        lvis_colors = list(
            itertools.islice(itertools.cycle(coco_metadata.stuff_colors), len(lvis_classes))
        )
        # rerrange to thing_classes, stuff_classes
        coco_thing_classes = coco_metadata.thing_classes
        coco_stuff_classes = [x for x in coco_metadata.stuff_classes if x not in coco_thing_classes]
        coco_thing_colors = coco_metadata.thing_colors
        coco_stuff_colors = [x for x in coco_metadata.stuff_colors if x not in coco_thing_colors]
        ade20k_thing_classes = ade20k_metadata.thing_classes
        ade20k_stuff_classes = [x for x in ade20k_metadata.stuff_classes if x not in ade20k_thing_classes]
        ade20k_thing_colors = ade20k_metadata.thing_colors
        ade20k_stuff_colors = [x for x in ade20k_metadata.stuff_colors if x not in ade20k_thing_colors]

        user_classes = []
        user_colors = [random_color(rgb=True, maximum=1) for _ in range(len(user_classes))]

        stuff_classes = coco_stuff_classes + ade20k_stuff_classes
        stuff_colors = coco_stuff_colors + ade20k_stuff_colors
        thing_classes = user_classes + coco_thing_classes + ade20k_thing_classes + lvis_classes
        thing_colors = user_colors + coco_thing_colors + ade20k_thing_colors + lvis_colors

        thing_dataset_id_to_contiguous_id = {x: x for x in range(len(thing_classes))}
        DatasetCatalog.register(
            "openvocab_dataset", lambda x: []
        )
        self.metadata = MetadataCatalog.get("openvocab_dataset").set(
            stuff_classes=thing_classes+stuff_classes,
            stuff_colors=thing_colors+stuff_colors,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        )
        #print("self.metadata:", self.metadata)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        self.predictor.set_metadata(self.metadata)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = OpenVocabVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5