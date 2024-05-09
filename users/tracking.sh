cd src
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --exp_id mix_mot17_half_dla34_ON_val_mot17




python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.35 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-35_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.45 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-45_ON_val_mot17 && \
python track.py mot --load_model ../models/mix_mot17_half_dla34.pth --conf_thres 0.5 --val_mot17 True --exp_id mix_mot17_half_dla34_nms-id1_th0-5_ON_val_mot17




