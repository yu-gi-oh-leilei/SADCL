# !/bin/bash
# cd ../../

# You can also select gamma_pos=0 and gamma_neg=0


# step 1
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc_supcon.py -a 'Q2L-R101-448' \
--dataset_dir '/media/data/maleilei/MLICdataset/' \
--backbone resnet101 --dataname coco14 --batch-size 64 --print-freq 400 \
--output './checkpoint/Q2L/ResNet_448_MSCOCO14/newattention_bs64_85' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3718 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 5e-5 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 5e-3 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9998 \
--seed 1 \
--lr_scheduler OneCycleLR \
--pattern_parameters single_lr \
--gpus 0,1


# setp 2
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc_supcon.py -a 'SADCL-R101-448' \
--dataset_dir '/media/data/maleilei/MLICdataset/' \
--backbone resnet101 --dataname coco14 --batch-size 64 --print-freq 10 \
--output './checkpoint/SADCL/ResNet_448_MSCOCO14/fixed_s2ploss/work2' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3718 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 5e-5 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 5e-3 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 --crop \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9998 \
--seed 1 \
--lr_scheduler OneCycleLR \
--pattern_parameters single_lr \
--resume './checkpoint/Q2L/ResNet_448_MSCOCO14/newattention_bs64_85/model_best.pth.tar' \
--prototype './data/prototype' \
--gpus 0,1