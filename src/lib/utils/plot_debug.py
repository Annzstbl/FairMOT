import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import cv2
from PIL import Image
from scipy.ndimage import zoom
import scipy.io as sio
import torch
import torch.nn.functional as F
import os

# for debug
def plot_heat(heat, filename='heat.png', norm=True):

    heat_data = heat[0,0,:,:].cpu().detach().numpy()
    # save npy
    sio.savemat('../debug/heat.mat', {'heat': heat_data})
    if norm:
        heat_data = (heat_data - np.min(heat_data)) / (np.max(heat_data) - np.min(heat_data))
        
    # cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
    cmap = 'jet'
    fig, ax = plt.subplots()
    cax = ax.imshow(heat_data, interpolation='nearest', cmap=cmap)
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Intensity')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    plt.savefig(os.path.join('../debug', filename), dpi=300)
    
def plot_heat_img(heat, img, filename='img_heat.png', norm=True):
    # 把热力图叠加到图像上显示
    heat_data = heat[0,0,:,:].cpu().detach().numpy()
    if norm:  
        heat_data = (heat_data - np.min(heat_data)) / (np.max(heat_data) - np.min(heat_data))
    
    heat_data = zoom(heat_data, (img.shape[0]/heat_data.shape[0], img.shape[1]/heat_data.shape[1]))
    fig, ax = plt.subplots()
    # cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
    cmap = 'jet'
    cax = ax.imshow(heat_data, cmap = cmap, interpolation='bilinear')
    ax.imshow(img, interpolation='nearest', alpha=0.5)
    
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Intensity')    
    
    
    plt.savefig(os.path.join('../debug', filename), dpi=300)
    
def plot_img(img, filename='img.png'):
    # img [width, height, 3]
    cv2.imwrite(os.path.join('../debug', filename), img)
    
def plot_dets(dets, img, th=0, filename='dets.png'):
    # 如果dets是tensor则转为numpy
    if isinstance(dets, torch.Tensor):
        dets = dets.cpu().detach().numpy()   
    
    # 删掉所有dets[:,4]小于th的行
    dets = dets[dets[:,4]>th]
    # dets [N, 5] 5: tlbr, score
    poses = dets[:,:4].astype(np.int32)
    scores = dets[:,4]
    
    for pos, score in zip(poses, scores):
        # to int
        cv2.rectangle(img, pt1=(pos[0], pos[1]), pt2=(pos[2], pos[3]), color=(0, 255, 0), thickness=2)
        # score
        cv2.putText(img, str(score), (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # center
        cv2.circle(img, (int((pos[0]+pos[2])/2), int((pos[1]+pos[3])/2)), 2, (0, 255, 0), 2)
        
    dst_filename = os.path.join('../debug', filename)
    cv2.imwrite(dst_filename, img)
    return img
        
        
def plot_dets_heat(dets, heat, img, filename='dets_heat.png'):
    # 把热力图叠加到heat上显示
    det_img = plot_dets(dets, img)
    heat_data = heat[0,0,:,:].cpu().detach().numpy()
    heat_data = (heat_data - np.min(heat_data)) / (np.max(heat_data) - np.min(heat_data))
    
    heat_data = zoom(heat_data, (det_img.shape[0]/heat_data.shape[0], det_img.shape[1]/heat_data.shape[1]))
    fig, ax = plt.subplots()
    ax.imshow(det_img, interpolation='nearest')
    # cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
    cmap = 'jet'
    cax = ax.imshow(heat_data, cmap = cmap, alpha=0.5, interpolation='bilinear')
    
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Intensity')    
    
    
    plt.savefig(os.path.join('../debug', filename), dpi=300)


def plot_cos_distance(id_feature, filename='cos_sim.png'):
    '''
        计算所有余弦距离,画成混淆矩阵图
        id_feature [N, C]
    '''
    
    # 如果是numpy,则转为tensor
    # 否则转到cpu
    if isinstance(id_feature, np.ndarray):
        id_feature = torch.tensor(id_feature)
    elif isinstance(id_feature, torch.Tensor):
        id_feature = id_feature.cpu()

    N, C = id_feature.shape
    norm_feature = F.normalize(id_feature, p=2, dim=1)
    cos_sim = torch.mm(norm_feature, norm_feature.t())
        
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # 值越大越蓝，越小越白，并且在方框上标明数值(保留两位小数)
    cax = ax.matshow(cos_sim, cmap='Blues')
    fig.colorbar(cax)
    num_rows, num_cols = cos_sim.shape
    for i in range(num_rows):
        for j in range(num_cols):
            ax.text(j, i, format(cos_sim[i, j], ".2f"), ha='center', va='center', color='black' if cos_sim[i, j] < 0.5 else 'white')
            
    dst_filename = os.path.join('../debug', filename)
    plt.savefig(dst_filename, dpi=300)