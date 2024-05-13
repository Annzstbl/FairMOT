cd src
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --exp_id mix_mot17_half_dla34_ON_val_mot17

# 废弃实验
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-35_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.45 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-45_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.5 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-5_ON_val_mot17

#nmstype 3
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.4 --exp_id mix_mot17_half_dla34_th04_nms-iou-04_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th04_nms-iou-05_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.6 --exp_id mix_mot17_half_dla34_th04_nms-iou-06_ON_val_mot17 
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.45 --exp_id mix_mot17_half_dla34_th04_nms-iou-045_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.55 --exp_id mix_mot17_half_dla34_th04_nms-iou-055_ON_val_mot17 
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th03_nms-iou-05_ON_val_mot17 
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th035_nms-iou-05_ON_val_mot17 
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.45 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th045_nms-iou-05_ON_val_mot17 
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.5 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th05_nms-iou-05_ON_val_mot17 

#nmstype 4
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.1 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-01_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.2 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-02_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.3 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-03_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.4 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-04_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.5 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-05_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.6 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-06_ON_val_mot17
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.7 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-07_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.8 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-08_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '4' --nms_iou_th 0.9 --exp_id mix_mot17_half_dla34_th03_nms-iou-id-09_ON_val_mot17
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '4' --nms_iou_th 0.7 --exp_id mix_mot17_half_dla34_th04_nms-iou-id-07_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '4' --nms_iou_th 0.8 --exp_id mix_mot17_half_dla34_th04_nms-iou-id-08_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '4' --nms_iou_th 0.9 --exp_id mix_mot17_half_dla34_th04_nms-iou-id-09_ON_val_mot17


#nmstype5
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '5' --nms_iou_th 0.7 --exp_id mix_mot17_half_dla34_th04_nms-iou-id-bi-07_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '5' --nms_iou_th 0.8 --exp_id mix_mot17_half_dla34_th04_nms-iou-id-bi-08_ON_val_mot17 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '5' --nms_iou_th 0.9 --exp_id mix_mot17_half_dla34_th04_nms-iou-id-bi-09_ON_val_mot17

### 以下名字第一次更新
#nmstype4
CUDA_VISIBLE_DEVICES=3 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --nms_type '4' --nms_iou_th 0.8 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id_th035_nmsth08 && \
CUDA_VISIBLE_DEVICES=3 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.45 --val_mot17 True --nms_type '4' --nms_iou_th 0.8 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id_th045_nmsth08 && \
CUDA_VISIBLE_DEVICES=3 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.5 --val_mot17 True --nms_type '4' --nms_iou_th 0.8 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id_th05_nmsth08

CUDA_VISIBLE_DEVICES=3 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.25 --val_mot17 True --nms_type '4' --nms_iou_th 0.7 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id_th025_nmsth07 && \
CUDA_VISIBLE_DEVICES=3 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --nms_type '4' --nms_iou_th 0.7 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id_th035_nmsth07 && \
CUDA_VISIBLE_DEVICES=3 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.45 --val_mot17 True --nms_type '4' --nms_iou_th 0.7 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id_th045_nmsth07 && \
CUDA_VISIBLE_DEVICES=3 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.5 --val_mot17 True --nms_type '4' --nms_iou_th 0.7 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id_th05_nmsth07



#nmstype5
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '5' --nms_iou_th 0.7 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-bi_th03_nmsth07 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '5' --nms_iou_th 0.8 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-bi_th03_nmsth08 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms_type '5' --nms_iou_th 0.9 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-bi_th03_nmsth09

CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --nms_type '5' --nms_iou_th 0.8 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-bi_th035_nmsth08 && \

#nmstype6
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.8 --nms_dis_th 5 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth08_disth5 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.8 --nms_dis_th 10 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth08_disth10 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.8 --nms_dis_th 15 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth08_disth15 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.8 --nms_dis_th 20 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth08_disth20 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.8 --nms_dis_th 25 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth08_disth25

CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.7 --nms_dis_th 5 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth07_disth5 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.7 --nms_dis_th 10 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth7_disth10 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.7 --nms_dis_th 15 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth7_disth15 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.7 --nms_dis_th 20 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth7_disth20 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '6' --nms_iou_th 0.7 --nms_dis_th 25 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-nei_th04_nmsth7_disth25


### 以下名字第二次更新
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.1 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh01_iouth05 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.2 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh02_iouth05 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.3 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh03_iouth05 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.4 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh04_iouth05 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.5 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh05_iouth05 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.6 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh06_iouth05 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.7 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh07_iouth05 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.8 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh08_iouth05 && \
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms_type '7' --nms_iou_th 0.5 --nms_cos_th 0.9 --exp_id mix-mot17-half-dla34_val-mot17_nms-iou-id-iou_th04_costh09_iouth05

# 自训参数
python track.py mot --load_model ../exp/mot/mix_mot17_half_dla34/model_last.pth --conf_thres 0.4 --val_mot17 True --exp_id mix_mot17_half_dla34_ON_val_mot17



####debug
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../exp/mot/mix_mot17_half_dla34/model_last.pth --conf_thres 0.4 --val_mot17 True --exp_id debug
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --nms_type '5' --nms_cos_th 0.8 --exp_id debug