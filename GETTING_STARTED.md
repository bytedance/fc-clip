## Getting Started with FC-CLIP

This document provides a brief intro of the usage of FC-CLIP.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Inference Demo with Pre-trained Models

We provide `demo.py` that is able to demo builtin configs. Run it with:
```
cd demo/
python demo.py \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


### Training & Evaluation in Command Line

We provide a script `train_net.py`, that is made to train all the configs provided in FC-CLIP.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md),
then run:
```
python train_net.py --num-gpus 8 \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml
```

The configs are made for 8-GPU training.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
To train on 1 GPU, you need to figure out learning rate and batch size by yourself:
```
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
```

To evaluate a model's performance, use
```
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `python train_net.py -h`.