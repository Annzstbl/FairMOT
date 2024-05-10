cd src
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --exp_id mix_mot17_half_dla34_ON_val_mot17




python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-35_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.45 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-45_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.5 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-5_ON_val_mot17


python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.4 --exp_id mix_mot17_half_dla34_th04_nms-iou-04_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th04_nms-iou-05_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.6 --exp_id mix_mot17_half_dla34_th04_nms-iou-06_ON_val_mot17 


python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.45 --exp_id mix_mot17_half_dla34_th04_nms-iou-045_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --nms-type '3' --nms-iou-th 0.55 --exp_id mix_mot17_half_dla34_th04_nms-iou-055_ON_val_mot17 

python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.3 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th03_nms-iou-05_ON_val_mot17 
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th035_nms-iou-05_ON_val_mot17 
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.45 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th045_nms-iou-05_ON_val_mot17 
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.5 --val_mot17 True --nms-type '3' --nms-iou-th 0.5 --exp_id mix_mot17_half_dla34_th05_nms-iou-05_ON_val_mot17 