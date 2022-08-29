#!/usr/bin
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node 1 --master_port=$RANDOM tools/train.py --config=polyp.BaFPR
