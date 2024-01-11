import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
from sklearn.metrics import roc_auc_score
from skimage import *
from sklearn import metrics




class FairDiceLoss(nn.Module):
    def __init__(self, n_classes, n_attr=3, gamma=0.):
        super(FairDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        self.n_attr = n_attr

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, attr, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())

        # overall_loss = 0.0
        # for i in range(0, self.n_classes):
        #     dice = self._dice_loss(inputs[:, i], target[:, i])
        #     overall_loss += dice 

        attr_set = np.unique(attr)
        tmp_weights = np.zeros(self.n_attr)
        for x in attr_set:
            tmp_input = inputs[attr == x]
            tmp_target = target[attr == x]
            tmp_loss = 0.
            for i in range(0, self.n_classes):
                dice = self._dice_loss(tmp_input[:, i], tmp_target[:, i])
                tmp_loss += dice 
            tmp_weights[x] = tmp_loss
        tmp_weights = (np.min(tmp_weights) / tmp_weights)**self.gamma
        
        # tmp_weights = (tmp_weights / np.max(tmp_weights))**self.gamma

        tmp_weights = torch.tensor(tmp_weights).cuda()
        tmp_weights = torch.tanh(tmp_weights)
       
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i]*tmp_weights[attr][:, None, None], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        return loss / self.n_classes


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        return dice, hd95, jaccard
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 1
    else:
        return 0, 0, 0
    
SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
    
# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    # outputs = outputs.unsqueeze(0)
    # labels = labels.unsqueeze(0)
    outputs = np.expand_dims(outputs, axis=0)
    labels = np.expand_dims(labels, axis=0)
    outputs[outputs > 0] = 1
    labels[labels > 0] = 1
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    # print("unthresholded_iou is {}".format(iou))

    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return iou  # Or thresholded.mean()  


def iou_numpy_v2(outputs: np.array, labels: np.array):
    # outputs = outputs.unsqueeze(0)
    # labels = labels.unsqueeze(0)
    outputs = np.expand_dims(outputs, axis=0)
    labels = np.expand_dims(labels, axis=0)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()  

def cdr_distance(outputs_cup: np.array, labels_cup: np.array, outputs_disc: np.array, labels_disc: np.array, ):
   
    pred_cup_num_pixels = np.sum(outputs_cup==1)
    pred_disc_num_pixels = np.sum(outputs_disc==1)

    gt_cup_num_pixels = np.sum(labels_cup==1)
    gt_disc_num_pixels = np.sum(labels_disc==1)
    
    cdr_dist = np.absolute(((pred_cup_num_pixels/pred_disc_num_pixels)-(gt_cup_num_pixels/gt_disc_num_pixels)))

    return cdr_dist  

def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != input_size[0] or y != input_size[1]:
                slice = zoom(slice, (input_size[0] / x, input_size[1] / y), order=3)  # previous using 0
            new_x, new_y = slice.shape[0], slice.shape[1]  # [input_size[0], input_size[1]]
            if new_x != patch_size[0] or new_y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / new_x, patch_size[1] / new_y), order=3)  # previous using 0, patch_size[0], patch_size[1]
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            net.eval()
            
            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0])
                output_masks = outputs['masks']
                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
        # only for debug
   
    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list


def compute_multi_class_auc(preds: np.array, labels: np.array):
    # Compute multi-class AUC
    auc_scores = []

    # Iterate through each class and compute the AUC
    for c in range(1, 3):
        binary_truth = (labels == c).astype(int)
        class_probs = preds[c, :, :]
        # Compute the AUC for the current class
        auc = roc_auc_score(binary_truth.ravel(), class_probs.ravel(), average="macro")
        # fpr, tpr, thresholds = metrics.roc_curve(binary_truth.ravel(), class_probs.ravel())
        # auc = metrics.auc(fpr, tpr)
        auc_scores.append(auc)

    # Compute the average AUC across all classes
    multi_class_auc = np.mean(auc_scores)
   
    return multi_class_auc


def equity_scaled_perf(identity_wise_perf, overall_perf, no_of_attr, alpha=1.):
    es_perf = 0
    tmp = 0
    
    for i in range(no_of_attr):
        one_attr_perf_list = identity_wise_perf[i]
        identity_perf = np.mean(one_attr_perf_list)
        tmp += np.abs(identity_perf-overall_perf)
    
    es_perf = (overall_perf / (alpha*tmp + 1))
    
    return es_perf

import statistics

def equity_scaled_std_perf(identity_wise_perf, overall_perf, no_of_attr, alpha=1.):
    es_perf = 0
    tmp = 0

    group_wise_perf = []
    for i in range(no_of_attr):
        one_attr_perf_list = identity_wise_perf[i]
        identity_perf = np.mean(one_attr_perf_list)
        group_wise_perf.append(identity_perf)
        # tmp += np.abs(identity_perf-overall_perf)
    stdev_value = statistics.stdev(group_wise_perf)    
    es_perf = (overall_perf / (alpha*stdev_value + 1))
    # es_auc = (overall_auc / (alpha*np.log(1+tmp) + 1))
    # es_auc = (overall_auc / (alpha*np.exp(tmp) + 1))

    return es_perf



def test_single_image(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, attr_label=0, idx=None):
    label = label.squeeze(0).cpu().detach().numpy()
    
    prediction = np.zeros_like(label)
    
    x, y = image.shape[2], image.shape[3]
   
    inputs = image.float().cuda()
    net.eval()

    with torch.no_grad():
        outputs = net(inputs, multimask_output, patch_size[0])
        output_masks = outputs['masks']

        softmaxed_prob = torch.softmax(output_masks, dim=1)
        auc_score = compute_multi_class_auc(softmaxed_prob.squeeze().cpu().detach().numpy(), label)
        # auc_score=0.0
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        
        out_h, out_w = out.shape
        if x != out_h or y != out_w:
            print('zoom prediction image')
            pred = zoom(out, (x / out_h, y / out_w), order=0)
        else:
            pred = out
        
        prediction = pred
     
    
    metric_list = []
    
    
    ### label == 1 is rim and label ==2 is cup and label >=1 is disc ######
    for i in range(1, classes + 1):
  
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        
    cdr_dist = cdr_distance(prediction==2, label==2, prediction>=1, label>=1)
    
    return metric_list, cdr_dist, auc_score
