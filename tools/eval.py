import os
import argparse
import tqdm
import sys
sys.path.append(os.getcwd())

import numpy as np
import cv2

#from PIL import Image
import imageio
from tabulate import tabulate

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils import StructureMeasure, original_WFb, EnhancedMeasure, Fmeasure_calu
#from utils.utils import *

#! taken from  https://github.com/plemeri/UACANet/blob/main/run/Eval.py and slightly modidied. Thanks for the authors.


def _args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, default='configs/UACANet-L.yaml')
    parser.add_argument('--pred_root', type=str, default=None)
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--gt_root', type=str, default=None)
    #parser.add_argument('--eval_datasets', type=list, default=None)
    return parser.parse_args()

def evaluate(args):
    datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    if os.path.isdir(args.result_path) is False:
        os.makedirs(args.result_path)

    method = os.path.split(args.pred_root)[-1]
    Thresholds = np.linspace(1, 0, 256)
    headers = ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm', 'maxDic', 'maxIoU', 'meanSen', 'maxSen', 'meanSpe', 'maxSpe']
    results = []
    
    if  True:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(datasets, desc='Expr - ' + method, total=len(
            datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        datasets = datasets

    for dataset in datasets:
        pred_root = os.path.join(args.pred_root, dataset)
        gt_root = os.path.join(args.gt_root, dataset, 'masks')

        preds = os.listdir(pred_root)
        gts = os.listdir(gt_root)

        preds.sort()
        gts.sort()

        threshold_Fmeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_Emeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_IoU = np.zeros((len(preds), len(Thresholds)))
        # threshold_Precision = np.zeros((len(preds), len(Thresholds)))
        # threshold_Recall = np.zeros((len(preds), len(Thresholds)))
        threshold_Sensitivity = np.zeros((len(preds), len(Thresholds)))
        threshold_Specificity = np.zeros((len(preds), len(Thresholds)))
        threshold_Dice = np.zeros((len(preds), len(Thresholds)))

        Smeasure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        MAE = np.zeros(len(preds))

        if  True:
            samples = tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(
                preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = enumerate(zip(preds, gts))

        for i, sample in samples:
            pred, gt = sample
            try:
                assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]
            except AssertionError:
                #cv2.resize(gt)
                #import pdb; pdb.set_trace()
                assert os.path.splitext(pred)[0].split('_')[-1] == os.path.splitext(gt)[0]
                #import pdb; pdb.set_trace()
                #print('hold for debugging')

            pred_mask = np.array(imageio.imread(os.path.join(pred_root, pred)))
            gt_mask = np.array(imageio.imread(os.path.join(gt_root, gt)))
            
            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]
            try:
                assert pred_mask.shape == gt_mask.shape
            except AssertionError:
                
                #print('hold for debugging')
                #import pdb; pdb.set_trace()
                gt_mask = cv2.resize(gt_mask, pred_mask.shape, interpolation=cv2.INTER_NEAREST)
                

            gt_mask = gt_mask.astype(np.float64) / 255
            gt_mask = (gt_mask > 0.5).astype(np.float64)

            pred_mask = pred_mask.astype(np.float64) / 255

            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

            threshold_E = np.zeros(len(Thresholds))
            threshold_F = np.zeros(len(Thresholds))
            threshold_Pr = np.zeros(len(Thresholds))
            threshold_Rec = np.zeros(len(Thresholds))
            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Spe = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))

            for j, threshold in enumerate(Thresholds):
                threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

                Bi_pred = np.zeros_like(pred_mask)
                Bi_pred[pred_mask >= threshold] = 1
                threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)
            
            threshold_Emeasure[i, :] = threshold_E
            threshold_Fmeasure[i, :] = threshold_F
            threshold_Sensitivity[i, :] = threshold_Rec
            threshold_Specificity[i, :] = threshold_Spe
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        result = []

        mae = np.mean(MAE)
        Sm = np.mean(Smeasure)
        wFm = np.mean(wFmeasure)

        column_E = np.mean(threshold_Emeasure, axis=0)
        meanEm = np.mean(column_E)
        maxEm = np.max(column_E)

        column_Sen = np.mean(threshold_Sensitivity, axis=0)
        meanSen = np.mean(column_Sen)
        maxSen = np.max(column_Sen)

        column_Spe = np.mean(threshold_Specificity, axis=0)
        meanSpe = np.mean(column_Spe)
        maxSpe = np.max(column_Spe)

        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)

        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)

        # result.extend([meanDic, meanIoU, wFm, Sm, meanEm, mae, maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe])
        # results.append([dataset, *result])
        
        out = []
        for metric in headers:
            out.append(eval(metric))

        result.extend(out)
        results.append([dataset, *result])

        csv = os.path.join(args.result_path, 'result_' + dataset + '.csv')
        if os.path.isfile(csv) is True:
            csv = open(csv, 'a')
        else:
            csv = open(csv, 'w')
            csv.write(', '.join(['method', *headers]) + '\n')

        out_str = method + ','
        for metric in result:
            out_str += '{:.4f}'.format(metric) + ','
        out_str += '\n'

        csv.write(out_str)
        csv.close()
    tab = tabulate(results, headers=['dataset', *headers], floatfmt=".3f")

    if True:
        print(tab)
        print("#"*20, "End Evaluation", "#"*20)
        
    return tab

if __name__ == "__main__":
    args = _args()
    
    evaluate(args)
