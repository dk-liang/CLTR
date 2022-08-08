#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 8218 train_distributed.py --gpu_id '0,1,2,3' \
--gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 1200 --lr 1e-4 \
--batch_size 16 --num_patch 1 --threshold 0.35 --test_per_epoch 20 --num_queries 700 \
--dataset nwpu --crop_size 256 --pre None --test_per_epoch 20  --test_patch --save