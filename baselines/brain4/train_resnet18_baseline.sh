#!/usr/bin/env bash
set -e

python3.10 tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet18.yaml \
  -o Global.device=gpu \
  -o Global.epochs=2 \
  -o Arch.class_num=4 \
  -o DataLoader.Train.dataset.image_root=/data \
  -o DataLoader.Train.dataset.cls_label_path=/data/train.txt \
  -o DataLoader.Eval.dataset.image_root=/data \
  -o DataLoader.Eval.dataset.cls_label_path=/data/val.txt \
  -o DataLoader.Train.sampler.batch_size=32 \
  -o DataLoader.Eval.sampler.batch_size=32 \
  -o DataLoader.Train.loader.num_workers=8 \
  -o DataLoader.Eval.loader.num_workers=8 \
  -o DataLoader.Train.loader.use_shared_memory=False \
  -o DataLoader.Eval.loader.use_shared_memory=False \
  -o Metric.Train[0].TopkAcc.topk=[1,2,4] \
  -o Metric.Eval[0].TopkAcc.topk=[1,2,4] \
  -o PostProcess.topk=4 \
  -o Optimizer.lr.learning_rate=0.025
