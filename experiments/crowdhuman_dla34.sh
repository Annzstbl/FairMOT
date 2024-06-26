cd src
CUDA_VISIBLE_DEVICES=0,1  python train.py mot --exp_id crowdhuman_dla34 --gpus 0,1 --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'

# resume
# CUDA_VISIBLE_DEVICES=0,1  python train.py mot --exp_id crowdhuman_dla34 --gpus 0,1 --batch_size 8 --resume --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'

cd ..