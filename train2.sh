CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --master_addr 127.0.0.1 --master_port 14523 \
    --nproc_per_node 4 --nnodes 1 --node_rank 0 \
    train.py \
    --data_path data/aloha/insert_flowers_bimanual_2 \
    --aug --aug_jitter --Tp 32 --Ta 32 --voxel_size 0.005 \
    --obs_feature_dim 512 --hidden_dim 512 \
    --nheads 8 --num_encoder_layers 4 --num_decoder_layers 1 \
    --dim_feedforward 2048 --dropout 0.1 \
    --ckpt_dir logs/aloha2/insert_flowers_bimanual_2 \
    --batch_size 240 --num_epochs 1000 --save_epochs 200 --num_workers 24 \
    --seed 233 \
    --norm_stat_filepath assets/norm_stat/insert_flowers_bimanual_2.pkl \
    --lr 5e-5