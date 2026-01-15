# Brain4 Baseline (ResNet18)

## Command
See `train_resnet18_baseline.sh`

## Notes
- 4 classes => set `Arch.class_num=4`, `PostProcess.topk=4`, and TopkAcc uses `[1,2,4]`
- Use `use_shared_memory=False` to avoid dataloader issues on some setups
- Dataset expected in container as:
  - /data/train.txt
  - /data/val.txt
  - images under /data/<class_name>/*.jpg|png
