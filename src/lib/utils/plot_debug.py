import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import cv2
from PIL import Image
from scipy.ndimage import zoom
import scipy.io as sio
import torch

# for debug
def plot_heat(heat):

    heat_data = heat[0,0,:,:].cpu().detach().numpy()
    # save npy
    sio.savemat('../debug/heat.mat', {'heat': heat_data})
    heat_data = (heat_data - np.min(heat_data)) / (np.max(heat_data) - np.min(heat_data))
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
    fig, ax = plt.subplots()
    cax = ax.imshow(heat_data, interpolation='nearest', cmap=cmap)
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Intensity')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.savefig('../debug/heat.png', dpi=300)
    
    
def plot_img(img):
    # img [width, height, 3]
    cv2.imwrite('../debug/img.png', img)
    
def plot_dets(dets, img, th=0):
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
    cv2.imwrite('../debug/dets.png', img)
    return img
        
        
def plot_dets_heat(dets, heat, img):
    # 把热力图叠加到heat上显示
    det_img = plot_dets(dets, img)
    heat_data = heat[0,0,:,:].cpu().detach().numpy()
    heat_data = (heat_data - np.min(heat_data)) / (np.max(heat_data) - np.min(heat_data))
    
    heat_data = zoom(heat_data, (det_img.shape[0]/heat_data.shape[0], det_img.shape[1]/heat_data.shape[1]))
    fig, ax = plt.subplots()
    ax.imshow(det_img, interpolation='nearest')
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
    cax = ax.imshow(heat_data, cmap = cmap, alpha=0.3, interpolation='bilinear')
    
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Intensity')    
    
    plt.savefig('../debug/dets_heat.png', dpi=300)
