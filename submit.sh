#!/bin/bash -x
#PJM -L rscgrp=cx-workshop
#PJM -L jobenv=singularity
#PJM -j
#PJM -S

module load singularity
singularity exec --bind /data/group1/${USER}:/data/group1/${USER} --nv ~/docker_images/BaFPR.sif \
bash train.sh