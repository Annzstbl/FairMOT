from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
import torch.nn.functional as F
from torchvision.ops import nms, box_iou
from typing import Optional

# 全局变量
IMG0_DEBUG_GLOBAL = None
DEBUG = False
    
def nms_decorator(nms_func):
    def wrapper(*args, **kwargs):
        # before
        output = nms_func(*args, **kwargs)
        # after   
        global DEBUG    
        if DEBUG:
            heat = output['heat']
            id_feature = output['id_feature']
            id_feature_bi = output['id_feature_bi']
            h, w = heat.shape[2:]
            # ################DEBUG
            heat = heat.view(h*w,1)
            id_feature = id_feature.permute([0,2,3,1]).view(h*w, -1)
            id_feature_bi = id_feature_bi.permute([0,2,3,1]).view(h*w, -1)

            heat_max = heat.argmax()
            cos_among_heatmax_idfeature = F.cosine_similarity(id_feature[heat_max:heat_max+1, :], id_feature, dim=1)#[h*w]
            cos_among_heatmax_idfeaturebi = F.cosine_similarity(id_feature_bi[heat_max:heat_max+1, :], id_feature_bi, dim=1)#[h*w]
            cos_betw_idfeature_idfeaturebi = F.cosine_similarity(id_feature, id_feature_bi, dim=1)#[h*w]
            
            from utils.plot_debug import plot_heat, plot_heat_img
            global IMG0_DEBUG_GLOBAL
            plot_heat(cos_among_heatmax_idfeature.view(1,1,h,w), 'cos_among_heatmax_idfeature.png', norm=False)
            plot_heat(cos_among_heatmax_idfeaturebi.view(1,1,h,w), 'cos_among_heatmax_idfeaturebi.png', norm=False)
            plot_heat(cos_betw_idfeature_idfeaturebi.view(1,1,h,w), 'cos_betw_idfeature_idfeaturebi.png', norm=False)
            plot_heat_img(cos_among_heatmax_idfeature.view(1,1,h,w), IMG0_DEBUG_GLOBAL, 'cos_among_heatmax_idfeature_img.png', norm=False)
            plot_heat_img(cos_among_heatmax_idfeaturebi.view(1,1,h,w), IMG0_DEBUG_GLOBAL, 'cos_among_heatmax_idfeaturebi_img.png', norm=False)
            plot_heat_img(cos_betw_idfeature_idfeaturebi.view(1,1,h,w), IMG0_DEBUG_GLOBAL, 'cos_betw_idfeature_idfeaturebi_img.png', norm=False)
            
        return output['heat']
    return wrapper

@nms_decorator
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return {'heat':heat * keep}

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

@nms_decorator
def _nms_id(heat, id_feature, kernel=5, alpha=0.1):
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
    
    return {'heat':heat * keep}
  
@nms_decorator    
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
    
    return {'heat':out_heat.view([1,1,height, width])}
    
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

@nms_decorator
def _nms_iou_id(heat, id_feature, cos_th):
    '''
        对heat进行基于id_feature余弦距离的nms操作
        (1) heat所有低于5e-2的全部删去
        (2) 对heat进行nms, 把nms的评价指标IOU换成id_feature余弦距离
    '''
    
    assert heat.shape[0] == 1
    assert heat.shape[1] == 1
    
    b, c, h, w = heat.size()
    _, c_id, _, _ = id_feature.size()
    
    heat = heat.view(h*w, 1)
    id_feature = id_feature.permute([0,2,3,1]).view(h*w, -1)
    inds = torch.arange(h*w).to(heat.device).view([h*w, 1])
    
    # 阈值5e-2
    remain = heat > 5e-2
    inds = inds[remain]
    heat_remain = heat[remain]
    
    # 排序
    sorted_heat, sort_inds = torch.sort(heat_remain, descending=True)
    inds = inds[sort_inds]
    
    #nms
    out_inds = []
    while len(inds) > 0: 
        out_inds.append(inds[0].view([1]))
        # inds = inds[1:] #取出第0个
        if len(inds) == 1:
            break
        cos_dis = F.cosine_similarity(id_feature[inds[0]:inds[0]+1,:], id_feature[inds,:], dim=1)
        remain = cos_dis < cos_th
        inds = inds[remain]
        if remain[0] :
            # 一般来讲iou[0] ==1 , 所以remain[0] == True, 这里以防万一
            inds = inds[1:]
    
    #把out_inds转为tensor
    out_inds = torch.cat(out_inds, dim = 0).view(-1) #[N]
    
    # 把keep的地方heat保留，其它的置0
    out_heat = torch.zeros_like(heat).view(-1)
    out_heat[out_inds] = heat.view(-1)[out_inds]

    return {'heat': out_heat.view([1,1,h, w]), 'id_feature': id_feature.permute([1,0]).view([1,-1, h, w])}

@nms_decorator
def _nms_iou_id_bi(heat, reg, id_feature, cos_th):
    '''
        对heat进行基于id_feature余弦距离的nms操作
        (1) heat所有低于5e-2的全部删去
        (2) 根据中心reg对id_feature进行插值修正
        (2) 对heat进行nms, 把nms的评价指标IOU换成id_feature余弦距离
        
        与函数_nms_iou_with_features不同的是,对id_feature进行了插值修正
    '''
    
    assert heat.shape[0] == 1
    assert heat.shape[1] == 1
    
    batch, c, height, width = heat.size()
    h = height
    w = width
    _, c_id, _, _ = id_feature.size()
    K = height*width
    
    inds = torch.arange(height*width).to(heat.device)
    ys = torch.true_divide(inds, width).int().float().unsqueeze(0) #[batch, height*width]行
    xs = (inds % width).int().float().unsqueeze(0)#[batch, height*width]列
   
    #找到中心点
    if reg is not None:
        reg = reg.permute([0,2,3,1]).view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    #插值id_feature  
    cent_pos = torch.cat([xs, ys], dim=2).view(batch, height, width, 2)
    # 归一化到[-1 -> 1]
    cent_pos = cent_pos / torch.tensor([width-1, height-1]).to(heat.device) * 2 - 1
    id_feature_bi = F.grid_sample(id_feature, cent_pos, mode='bilinear', align_corners=True)
    
    # 与_nms_iou_with_features相同        
    heat = heat.view(h*w, 1)
    id_feature_bi = id_feature_bi.permute([0,2,3,1]).view(h*w, -1)
    inds = torch.arange(h*w).to(heat.device).view([h*w, 1])
    

    # 阈值5e-2
    remain = heat > 5e-2
    inds = inds[remain]
    heat_remain = heat[remain]
    
    # 排序
    sorted_heat, sort_inds = torch.sort(heat_remain, descending=True)
    inds = inds[sort_inds]
    
    #nms
    out_inds = []
    while len(inds) > 0: 
        out_inds.append(inds[0].view([1]))
        # inds = inds[1:] #取出第0个
        if len(inds) == 1:
            break
        iou = F.cosine_similarity(id_feature_bi[inds[0]:inds[0]+1,:], id_feature_bi[inds,:], dim=1)
        remain = iou < cos_th
        inds = inds[remain]
        if remain[0] :
            # 一般来讲iou[0] ==1 , 所以remain[0] == True, 这里以防万一
            inds = inds[1:]
    
    #把out_inds转为tensor
    out_inds = torch.cat(out_inds, dim = 0).view(-1) #[N]
    
    # 把keep的地方heat保留，其它的置0
    out_heat = torch.zeros_like(heat).view(-1)
    out_heat[out_inds] = heat.view(-1)[out_inds]  
    
    return {'heat': out_heat.view([1,1,h, w]), 'id_feature': id_feature, 'id_feature_bi': id_feature_bi.permute([1,0]).view([1,-1, h, w])}

@nms_decorator
def _nms_iou_id_nei(heat, id_feature, cos_th, dis_th):
    '''
        对heat进行基于id_feature余弦距离的nms操作
        (1) heat所有低于5e-2的全部删去
        (2) 对heat进行nms, 把nms的评价指标IOU换成id_feature余弦距离
        (3) 只对距离小于dis_th的生效
    '''
       
    assert heat.shape[0] == 1
    assert heat.shape[1] == 1
    
    b, c, h, w = heat.size()
    _, c_id, _, _ = id_feature.size()
    
    heat = heat.view(h*w, 1)
    id_feature = id_feature.permute([0,2,3,1]).view(h*w, -1)
    inds = torch.arange(h*w).to(heat.device).view([h*w, 1])
    
    # 阈值5e-2
    remain = heat > 5e-2
    inds = inds[remain]
    heat_remain = heat[remain]
    
    # 排序
    sorted_heat, sort_inds = torch.sort(heat_remain, descending=True)
    inds = inds[sort_inds] # [N, 1]
    coor = torch.cat([torch.true_divide(inds, w).int().float().view(-1,1), (inds % w).int().float().view(-1,1)], dim=1) #[N, 2]
    
    #nms
    out_inds = []
    while len(inds) > 0: 
        out_inds.append(inds[0].view([1]))
        # inds = inds[1:] #取出第0个
        if len(inds) == 1:
            break
        iou = F.cosine_similarity(id_feature[inds[0]:inds[0]+1,:], id_feature[inds,:], dim=1)
        # 求coor[0]和其它coor的L2距离
        dis = torch.sqrt(torch.sum((coor[0] - coor)**2, dim=1))#[ N, 1]
        
        remain = (iou < cos_th) | (dis > dis_th)
        inds = inds[remain]
        coor = coor[remain]
        if remain[0] :
            # 一般来讲iou[0] ==1 , 所以remain[0] == True, 这里以防万一
            inds = inds[1:]
    
    #把out_inds转为tensor
    out_inds = torch.cat(out_inds, dim = 0).view(-1) #[N]
    
    # 把keep的地方heat保留，其它的置0
    out_heat = torch.zeros_like(heat).view(-1)
    out_heat[out_inds] = heat.view(-1)[out_inds]

    return {'heat': out_heat.view([1,1,h, w]), 'id_feature': id_feature.permute([1,0]).view([1,-1, h, w])}

@nms_decorator
def _nms_iou_id_iou(heat, all_bboxes, id_feature, cos_th, iou_th):
    '''
        对heat进行基于id_feature余弦距离的nms操作
        (1) heat所有低于5e-2的全部删去
        (2) 对heat进行nms, 
        (3) 评价指标一: cos距离小于cos_th的保留
        (4) 评价指标二: iou距离小于iou_th的保留
    '''
       
    assert heat.shape[0] == 1
    assert heat.shape[1] == 1
    
    b, c, h, w = heat.size()
    _, c_id, _, _ = id_feature.size()
    
    heat = heat.view(h*w, 1)
    id_feature = id_feature.permute([0,2,3,1]).view(h*w, -1)
    inds = torch.arange(h*w).to(heat.device).view([h*w, 1])
    all_bboxes = all_bboxes.view([h*w, -1]) #[hw, 6]
    
    # 阈值5e-2
    remain = heat > 5e-2
    inds = inds[remain]
    heat_remain = heat[remain]
    
    # 排序
    sorted_heat, sort_inds = torch.sort(heat_remain, descending=True)
    inds = inds[sort_inds] # [N, 1]
    
    #nms
    out_inds = []
    while len(inds) > 0: 
        out_inds.append(inds[0].view([1]))
        # inds = inds[1:] #取出第0个
        if len(inds) == 1:
            break
        cos_dis = F.cosine_similarity(id_feature[inds[0]:inds[0]+1,:], id_feature[inds,:], dim=1)
        iou_dis = box_iou(all_bboxes[inds[0]:inds[0]+1,:4], all_bboxes[inds,:4]).view(-1)#[N]
        
        # nms-iou是 iou_dis< iou_th的全部保留
        # 在此基础上, 额外保留一些cos_dis < cos_th的
        # 表示如果被iou剔除掉了, 但是cos距离足够小表示不是同一目标，不应被剔除
        remain = (iou_dis < iou_th) | (cos_dis < cos_th)
        inds = inds[remain]
        
        if remain[0] :
            # 一般来讲iou[0] ==1 , 所以remain[0] == True, 这里以防万一
            inds = inds[1:]
    
    #把out_inds转为tensor
    out_inds = torch.cat(out_inds, dim = 0).view(-1) #[N]
    
    # 把keep的地方heat保留，其它的置0
    out_heat = torch.zeros_like(heat).view(-1)
    out_heat[out_inds] = heat.view(-1)[out_inds]

    return {'heat': out_heat.view([1,1,h, w]), 'id_feature': id_feature.permute([1,0]).view([1,-1, h, w])}
    
    
def nms_factory(heat, wh, reg, id_feature, ltrb, all_bboxes, nmsopt):
    nms_type = nmsopt['type']
    param = nmsopt['param']
    # 临时删掉type键，用于输入到nms函数中
    if nms_type == '1':
        return _nms(heat, **param)
    elif nms_type == '2':
        return _nms_id(heat, id_feature, **param)
    elif nms_type =='3':
        return _nms_iou(heat, all_bboxes, **param)
    elif nms_type =='4':
        return _nms_iou_id(heat, id_feature, **param)
    elif nms_type=='5':
        return _nms_iou_id_bi(heat, reg, id_feature, **param)
    elif nms_type =='6':
        return _nms_iou_id_nei(heat,  id_feature, **param)
    elif nms_type == '7':
        return _nms_iou_id_iou(heat, all_bboxes, id_feature, **param)
          
def mot_decode(heat, wh, id_feature, nmsopt, reg=None, ltrb=False, K=100, img0_debug=None):
    '''
        nmsopt: nms参数
        
        
        output:
            detections: [b, K, 6]
            inds: [b, K]
        
    '''
    global IMG0_DEBUG_GLOBAL
    IMG0_DEBUG_GLOBAL = img0_debug
    
    batch, cat, height, width = heat.size()
    all_bboxes = _get_all_bboxes(heat, wh, reg, ltrb)#[1, height*width, 6]
    heat = nms_factory(heat, wh, reg, id_feature, ltrb, all_bboxes, nmsopt)# nms
   
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



