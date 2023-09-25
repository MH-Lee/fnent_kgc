DATA_DIR=../datasets

MODEL_NAME=FNetE
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LOSS=Cross_Entropy_Loss
TRAIN_BS=512
EVAL_BS=256
DIM=256
DIM_FEEDFORWARD=1024
LEARNING_RATE=0.003
MAX_EPOCHES=1000
REGULARIZATION=0
NUM_WORKERS=4
NUM_BLOCKS=3
CHECK_PER_EPOCH=30
LITMODEL_NAME=FNetELitModel
TRAIN_SAMPLER_CLASS=ConvSampler
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --loss $LOSS \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --emb_dim $DIM \
    --lr $LEARNING_RATE \
    --max_epochs $MAX_EPOCHES \
    --regularization $REGULARIZATION \
    --num_workers $NUM_WORKERS \
    --check_per_epoch $CHECK_PER_EPOCH \
    --litmodel_name $LITMODEL_NAME \
    --nblocks $NUM_BLOCKS \
    --dim_feedforward $DIM_FEEDFORWARD \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --use_wandb \
    --save_config