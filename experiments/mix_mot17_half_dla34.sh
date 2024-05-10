cd src
CUDA_VISIBLE_DEVICES=0,1 python train.py mot --exp_id mix_mot17_half_dla34 --gpus 0,1 --num_workers 16 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/data_half.json' 
cd ..