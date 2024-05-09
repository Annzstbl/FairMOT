from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
import torch.nn.functional as F

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = torch.true_divide(topk_inds, width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = torch.true_divide(topk_inds, width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.true_divide(topk_ind, K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

# 这部分是通过heatmap得到检测框的代码
def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)#这里的nms是，如果一个点是kernel内最大的，则保留

    scores, inds, clses, ys, xs = _topk(heat, K=K)# 选取前K个点
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds




def _nms_with_features(heat, id_feature, kernel=5):
    # heat: [1, 1, w, h], id_feature:[1, 128, w, h]
    # kernel: 窗口大小
    # 按以下步骤进行：
    # 计算每个窗口内所有点和中心点的id_feature的余弦距离，将其叠加到heat上
    # 判断叠加之后，将该点的值变为窗口内最大的值
    # todo
    pad = (kernel - 1) // 2
    _, C, H, W = id_feature.shape
    
    padded_heat = F.pad(heat, (pad, pad, pad, pad) )
    padded_id_feature = F.pad(id_feature, (pad, pad, pad, pad) )
    
    # 使用unfold提取所有窗口的特征 [1, C, H, W, kernel*kernel]
    windows_feature = padded_id_feature.unfold(2, kernel, 1).unfold(3, kernel, 1).contiguous().view(1, C, H, W, kernel*kernel)
    
    # 提取中心特征，[1, 128, H, W,]
    center_features = id_feature.unsqueeze(-1)
    
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(windows_feature, center_features, dim=1)# [1, H, W, kernel*kernel]
    
    # 提取heat窗口 [1, 1, H, W, kernel*kernel]
    windows_heat = padded_heat.unfold(2, kernel, 1).unfold(3, kernel, 1).contiguous().view(1, H, W, kernel*kernel)
    
    # cos_sim做一次乘方
    cos_sim = cos_sim ** 0.1
    
    # 点乘
    cos_heat = windows_heat * cos_sim
    
    # 取最大值
    max_heat = cos_heat.max(dim=-1, keepdim=False)[0].unsqueeze(1)
    
    keep = (max_heat == heat).float()
    
    return heat * keep   
    
def mot_decode_new (heat, wh, id_feature, reg=None, ltrb=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat_one = _nms_with_features(heat, id_feature)#这里的nms是，如果一个点是kernel内最大的，则保留
    heat_two = _nms(heat, 5)
    ########################### debug
    # 比较heat_one和heat_two
    count_one = torch.count_nonzero(heat_one)
    count_two = torch.count_nonzero(heat_two)
    # 找到在heat_one中的点，但是不在heat_two中的点
    # and 在heat_two中的点，但是不在heat_one中的点
    one_dec_two = heat_one- heat_two
    two_dec_one = heat_two - heat_one
    one_dec_two = one_dec_two>0
    two_dec_one = two_dec_one>0
    in_one_out_two = torch.count_nonzero(one_dec_two)
    in_two_out_one = torch.count_nonzero(two_dec_one)
    
    heat=heat_two
    

    scores, inds, clses, ys, xs = _topk(heat, K=K)# 选取前K个点
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds

