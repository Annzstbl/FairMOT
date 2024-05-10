from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
import torch.nn.functional as F
from torchvision.ops import nms
from typing import Optional

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

    topk_inds = topk_inds % (height * width)#topk_inds[b,num_class,K]
    # 左上起点的 ys:行 xs:列
    topk_ys   = torch.true_divide(topk_inds, width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)#topk_ind->(1,2,3,4,...,49)
    topk_clses = torch.true_divide(topk_ind, K).int()#都是0
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)# topk_inds.view(batch, -1, 1)->[b, num_class*K, 1]
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _nms_with_features(heat, id_feature, kernel=5, alpha=0.1):
    # heat: [1, 1, w, h], id_feature:[1, 128, w, h]
    # kernel: 窗口大小
    # 按以下步骤进行：
    # 计算每个窗口内所有点和中心点的id_feature的余弦距离，将其叠加到heat上
    # 判断叠加之后，将该点的值变为窗口内最大的值
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
    cos_sim = cos_sim ** alpha
    
    # 点乘
    cos_heat = windows_heat * cos_sim
    
    # 取最大值
    max_heat = cos_heat.max(dim=-1, keepdim=False)[0].unsqueeze(1)
    
    keep = (max_heat == heat).float()
    
    return heat * keep   
      
def _nms_iou(heat, all_bboxes, iou_th):
    '''
        heat: [b, c, h, w] -> [1,1,h,w]
        all_bboxes: [b, num, 6]
        
        输出 ind
    '''
    assert heat.shape[0] == 1
    assert heat.shape[1] == 1
    batch, c, height, width = heat.size()
    all_bboxes = all_bboxes.squeeze() #[num, 6]
    inds = torch.arange(height* width).to(heat.device)
    remain = all_bboxes[:,4]> 5e-2
    all_bboxes = all_bboxes[remain]
    inds = inds[remain]
    keep_ind = nms(boxes=all_bboxes[:,:4], scores=all_bboxes[:,4], iou_threshold=iou_th)#[b, num]
    
    # 把keep的地方heat保留，其它的置0
    out_heat = torch.zeros_like(heat).view(-1)
    out_heat[inds[keep_ind]] = heat.view(-1)[inds[keep_ind]]
    return out_heat.view([1,1,height, width])
    
def _get_all_bboxes(heat: torch.tensor, wh:torch.tensor, reg: Optional[torch.tensor] =None, ltrb:bool = False):
    '''
        只支持单分类
        heat:[b,c,h,w]
        wh:[b,2|4, h, w]
        reg:[b, 2, h, w] or None
    '''
    batch, c, height, width = heat.size()
    assert c == 1
    K = height*width
    inds = torch.arange(height*width).to(heat.device)
    ys = torch.true_divide(inds, width).int().float().unsqueeze(0) #[batch, height*width]
    xs = (inds % width).int().float().unsqueeze(0)#[batch, height*width]
    if reg is not None:
        reg = reg.permute([0,2,3,1]).view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    
    if ltrb:
        wh = wh.permute([0,2,3,1]).view(batch, K, 4)
    else:
        wh = wh.permute([0,2,3,1]).view(batch, K, 2)
        
    #bboxes [batch, K, 4]
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
    
    scores = heat.view(batch, K, 1)  
    clses = torch.zeros_like(scores)#单分类
    return torch.cat([bboxes, scores, clses], dim=2)   

def nms_factory(heat, wh, id_feature, ltrb, all_bboxes, nmsopt):
    nms_type = nmsopt['type']
    param = nmsopt['param']
    # 临时删掉type键，用于输入到nms函数中
    if nms_type == '1':
        return _nms(heat, **param)
    elif nms_type == '2':
        return _nms_with_features(heat, id_feature, **param)
    elif nms_type =='3':
        return _nms_iou(heat, all_bboxes, **param)
    
def mot_decode(heat, wh, id_feature, nmsopt, reg=None, ltrb=False, K=100, img0_debug=None):
    '''
        nmsopt: nms参数
        
        
        output:
            detections: [b, K, 6]
            inds: [b, K]
        
    '''
    batch, cat, height, width = heat.size()
    all_bboxes = _get_all_bboxes(heat, wh, reg, ltrb)#[1, height*width, 6]
    heat = nms_factory(heat, wh, id_feature, ltrb, all_bboxes, nmsopt)# nms
   
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
