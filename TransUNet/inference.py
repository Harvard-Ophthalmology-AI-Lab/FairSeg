import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_fairseg import FairSeg_dataset, TestGenerator
from utils import test_single_volume, test_single_image, equity_scaled_perf, equity_scaled_std_perf
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from fairlearn.metrics import *

class_to_name = {1: 'cup', 2: 'disc'}

# attr_to_race = {2: 0, 3: 1, 7:2}
# attr_to_language = {0: 0, 1: 1, 2:2, -1:-1}

def inference(args, model, db_config, test_save_path=None, no_of_attr=3):
    db_test = db_config['Dataset'](base_dir=args.datadir, args=args, split='test', attr_label=args.attribute, \
                                    transform=transforms.Compose([TestGenerator(output_size=[args.img_size, args.img_size], \
                                                                             low_res=[224, 224], center_crop_size=args.center_crop_size, use_normalize=True)]))
    multimask_output=None
    # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    
    model.eval()
    metric_list = 0.0
    nsd_list = 0.0
    asd_list = 0.0
    hausdorff_list = 0.0

    cdr_overall = 0.0
    auc_overall = 0.0

    cdr_by_attr = [ [] for _ in range(no_of_attr) ]
    
    auc_by_attr = [ [] for _ in range(no_of_attr) ]

    dice_by_attr = [ [] for _ in range(no_of_attr) ]
    hd_by_attr = [ [] for _ in range(no_of_attr) ]
    jc_by_attr = [ [] for _ in range(no_of_attr) ]

    NSD_by_attr = [ [] for _ in range(no_of_attr) ]
    ASD_by_attr = [ [] for _ in range(no_of_attr) ]
    Hausdorff_by_attr = [ [] for _ in range(no_of_attr) ]

    dice_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    hd_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    jc_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    NSD_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    ASD_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    Hausdorff_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    
    
    dice_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    hd_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    jc_by_attr_rim = [ [] for _ in range(no_of_attr) ]

    NSD_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    ASD_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    Hausdorff_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    
    all_preds_rim = []
    all_gts_rim = []
    all_attrs_rim = []
     
    all_preds_cup = []
    all_gts_cup = []
    all_attrs_cup = []

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name, attr_label = sampled_batch['image'], sampled_batch['label'], \
            sampled_batch['pid'], sampled_batch['attr_label']
        metric_i, cdr_dist, auc_score, nsd_metric, asd_metric, hausdorff_metric, \
            preds_array, gts_array, attrs_array  = test_single_image(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, \
                                        attr_label=attr_label, idx=i_batch)
        
        all_preds_rim.append(preds_array[0]) 
        all_gts_rim.append(gts_array[0]) 
        all_attrs_rim.append(attrs_array[0])
        
        all_preds_cup.append(preds_array[1]) 
        all_gts_cup.append(gts_array[1]) 
        all_attrs_cup.append(attrs_array[1])

        metric_list += np.array(metric_i)
        nsd_list += np.array(nsd_metric)
        asd_list += np.array(asd_metric)

        auc_overall += auc_score
        cdr_overall += cdr_dist

        attr_label = attr_label.detach().cpu().numpy().item()

        if attr_label != -1:
            dice_by_attr[attr_label].append(np.mean(metric_i, axis=0)[0])
            hd_by_attr[attr_label].append(np.mean(metric_i, axis=0)[1])
            jc_by_attr[attr_label].append(np.mean(metric_i, axis=0)[2])
            cdr_by_attr[attr_label].append(cdr_dist)
            auc_by_attr[attr_label].append(auc_score)

            NSD_by_attr[attr_label].append(np.mean(nsd_metric))
            ASD_by_attr[attr_label].append(np.mean(asd_metric))

            # compute for rim 
            dice_by_attr_rim[attr_label].append(metric_i[0][0])
            hd_by_attr_rim[attr_label].append(metric_i[0][1])
            jc_by_attr_rim[attr_label].append(metric_i[0][2])

            NSD_by_attr_rim[attr_label].append(nsd_metric[0])
            ASD_by_attr_rim[attr_label].append(asd_metric[0])

            # compute for cup 
            dice_by_attr_cup[attr_label].append(metric_i[1][0])
            hd_by_attr_cup[attr_label].append(metric_i[1][1])
            jc_by_attr_cup[attr_label].append(metric_i[1][2])
            
            NSD_by_attr_cup[attr_label].append(nsd_metric[1])
            ASD_by_attr_cup[attr_label].append(asd_metric[1])

    
    metric_list = metric_list / len(db_test)
    mean_auc = auc_overall / len(db_test)
    mean_cdr_dist = cdr_overall / len(db_test) 
    
    nsd_list = nsd_list / len(db_test)
    asd_list = asd_list / len(db_test)
    
    # print(metric_list.shape)
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_nsd = np.mean(nsd_list)
    mean_asd = np.mean(asd_list)

    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jaccard = np.mean(metric_list, axis=0)[2]
    cup_overall_dice = metric_list[1][0]
    rim_overall_dice = metric_list[0][0]
    
    rim_overall_nsd = nsd_list[0][0][0]
    cup_overall_nsd = nsd_list[1][0][0]
    
    rim_overall_asd = asd_list[0]
    cup_overall_asd = asd_list[1]
    
    cup_overall_hd95 = metric_list[1][1]
    rim_overall_hd95 = metric_list[0][1]

    cup_overall_jaccard = metric_list[1][2]
    rim_overall_jaccard = metric_list[0][2]
    
    
    logging.info('--------- Overall Performance for Attribute: {} -----------'.format(args.attribute))

    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr[one_attr]
        one_attr_hd_list = hd_by_attr[one_attr]
        one_attr_jc_list = jc_by_attr[one_attr]
        one_attr_auc_list = auc_by_attr[one_attr]
        one_attr_cdr_list = cdr_by_attr[one_attr]
        
        one_attr_nsd_list = NSD_by_attr[one_attr]
        one_attr_asd_list = ASD_by_attr[one_attr]

        logging.info(f'{one_attr}-attr overall dice: {np.mean(one_attr_dice_list):.4f}')
        logging.info(f'{one_attr}-attr overall hd95: {np.mean(one_attr_hd_list):.4f}')
        logging.info(f'{one_attr}-attr overall Jaccard/IoU: {np.mean(one_attr_jc_list):.4f}')

        logging.info(f'{one_attr}-attr overall NSD: {np.mean(one_attr_nsd_list):.4f}')
        logging.info(f'{one_attr}-attr overall ASD: {np.mean(one_attr_asd_list):.4f}')
        # logging.info(f'{one_attr}-attr overall AUC: {np.mean(one_attr_auc_list):.4f}')
        # logging.info(f'{one_attr}-attr overall CDR Distance: {np.mean(one_attr_cdr_list):.4f}')



    logging.info('--------- Cup Performance for Attribute: {} -----------'.format(args.attribute))
    
    logging.info(f'Cup Overall Dice: {cup_overall_dice:.4f}')
    logging.info(f'Cup Overall hd95: {cup_overall_hd95:.4f}')
    logging.info(f'Cup Overall IoU: {cup_overall_jaccard:.4f}')
    
    logging.info(f'Cup Overall NSD: {cup_overall_nsd:.4f}')
    logging.info(f'Cup Overall ASD: {cup_overall_asd:.4f}')
    
    es_cup_dice = equity_scaled_perf(dice_by_attr_cup, cup_overall_dice, no_of_attr)
    es_std_cup_dice = equity_scaled_std_perf(dice_by_attr_cup, cup_overall_dice, no_of_attr)
    es_cup_iou = equity_scaled_perf(jc_by_attr_cup, cup_overall_jaccard,no_of_attr)
    es_std_cup_iou= equity_scaled_std_perf(jc_by_attr_cup, cup_overall_jaccard,no_of_attr)
    
    logging.info(f'Cup Es-Dice: {es_cup_dice:.4f}')
    logging.info(f'Cup Es-IoU: {es_cup_iou:.4f}')
    
    logging.info(f'Cup Es-Std-Dice: {es_std_cup_dice:.4f}')
    logging.info(f'Cup Es-Std-IoU: {es_std_cup_iou:.4f}')

    
    pred_cup_array = np.concatenate(all_preds_cup).flatten()
    gts_cup_array = np.concatenate(all_gts_cup).flatten()
    attr_cup_array = np.concatenate(all_attrs_cup).flatten()
    
    dpd_cup = demographic_parity_difference(gts_cup_array,
                                    pred_cup_array,
                                    sensitive_features=attr_cup_array)
    
    eod_cup = equalized_odds_difference(gts_cup_array,
                                   pred_cup_array,
                                    sensitive_features=attr_cup_array)
    logging.info(f'Cup DPD: {dpd_cup:.4f}')
    logging.info(f'Cup DEodds: {eod_cup:.4f}')
    # dpd_rim, eod_rim, dpd_cup, eod_cup
    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr_cup[one_attr]
        one_attr_hd_list = hd_by_attr_cup[one_attr]
        one_attr_jc_list = jc_by_attr_cup[one_attr]
        
        logging.info(f'{one_attr}-attr dice for Cup: {np.mean(one_attr_dice_list):.4f}')
        logging.info(f'{one_attr}-attr hd95 for Cup: {np.mean(one_attr_hd_list):.4f}')
        logging.info(f'{one_attr}-attr Jaccard/IoU for Cup: {np.mean(one_attr_jc_list):.4f}')

        one_attr_nsd_list = NSD_by_attr_cup[one_attr]
        one_attr_asd_list = ASD_by_attr_cup[one_attr]

        logging.info(f'{one_attr}-attr NSD for cup: {np.mean(one_attr_nsd_list):.4f}')
        logging.info(f'{one_attr}-attr ASD for cup: {np.mean(one_attr_asd_list):.4f}')
  
    logging.info('--------- Rim Performance for Attribute: {} -----------'.format(args.attribute))
    
     
    logging.info(f'Rim Overall Dice: {rim_overall_dice:.4f}')
    logging.info(f'Rim Overall hd95: {rim_overall_hd95:.4f}')
    logging.info(f'Rim Overall IoU: {rim_overall_jaccard:.4f}')

    logging.info(f'Rim Overall NSD: {rim_overall_nsd:.4f}')
    logging.info(f'Rim Overall ASD: {rim_overall_asd:.4f}')

    es_rim_dice = equity_scaled_perf(dice_by_attr_rim, rim_overall_dice, no_of_attr)
    es_std_rim_dice = equity_scaled_std_perf(dice_by_attr_rim, rim_overall_dice, no_of_attr)
    es_rim_iou = equity_scaled_perf(jc_by_attr_rim, rim_overall_jaccard, no_of_attr)
    es_std_rim_iou = equity_scaled_std_perf(jc_by_attr_rim, rim_overall_jaccard, no_of_attr)

    logging.info(f'Rim Es-Dice: {es_rim_dice:.4f}')
    logging.info(f'Rim Es-IoU: {es_rim_iou:.4f}')

    logging.info(f'Rim Es-Std-Dice: {es_std_rim_dice:.4f}')
    logging.info(f'Rim Es-Std-IoU: {es_std_rim_iou:.4f}')

    pred_rim_array = np.concatenate(all_preds_rim).flatten()
    gts_rim_array = np.concatenate(all_gts_rim).flatten()
    attr_rim_array = np.concatenate(all_attrs_rim).flatten()
    
    dpd_rim = demographic_parity_difference(gts_rim_array,
                                    pred_rim_array,
                                    sensitive_features=attr_rim_array)
        
    eod_rim = equalized_odds_difference(gts_rim_array,
                                    pred_rim_array,
                                    sensitive_features=attr_rim_array)
    
    logging.info(f'Rim DPD: {dpd_rim:.4f}')
    logging.info(f'Rim DEodds: {eod_rim:.4f}')
    # dpd_rim, eod_rim, dpd_cup, eod_cup

    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr_rim[one_attr]
        one_attr_hd_list = hd_by_attr_rim[one_attr]
        one_attr_jc_list = jc_by_attr_rim[one_attr]

        logging.info(f'{one_attr}-attr dice for Rim: {np.mean(one_attr_dice_list):.4f}')
        logging.info(f'{one_attr}-attr hd95 for Rim: {np.mean(one_attr_hd_list):.4f}')
        logging.info(f'{one_attr}-attr Jaccard/IoU for Rim: {np.mean(one_attr_jc_list):.4f}')

        one_attr_nsd_list = NSD_by_attr_rim[one_attr]
        one_attr_asd_list = ASD_by_attr_rim[one_attr]

        logging.info(f'{one_attr}-attr NSD for rim: {np.mean(one_attr_nsd_list):.4f}')
        logging.info(f'{one_attr}-attr ASD for rim: {np.mean(one_attr_asd_list):.4f}')

    logging.info('------------------------------------------------------')
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jaccard : %f, mean_auc : %f, mean_cdr_distance : %f ' \
                 % (performance, mean_hd95, mean_jaccard, mean_auc, mean_cdr_dist))
    logging.info("Testing Finished!")

    return 1