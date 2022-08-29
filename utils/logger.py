import torch
from torch.utils.tensorboard import SummaryWriter
# set protobuf==3.20.1

import os
import logging
import datetime

def _get_logger(log_dir: str):
    #print(log_dir)
    #assert os.path.exists(log_dir), print(f'not able to locate the log dir {log_dir}.')
    # using tensorboard logger
    writer = SummaryWriter(log_dir=log_dir, filename_suffix='.metrics')
    img_writer = SummaryWriter(log_dir=log_dir, filename_suffix='.tensorboardimgs')

    # using INFO logger
    logger = logging.getLogger('info')
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    logger_path = log_dir + f'/logger_starts_at_{ts}.log'
    hdlr = logging.FileHandler(logger_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    return writer, img_writer, logger
