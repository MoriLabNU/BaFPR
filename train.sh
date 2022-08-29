#!/usr/bin
CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node 3 --master_port=$RANDOM tools/train.py --config=polyp.BaFPR
