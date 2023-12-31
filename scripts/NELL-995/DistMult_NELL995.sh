DATA_DIR=../datasets

MODEL_NAME=DistMult
DATASET_NAME=NELL-995
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
MAX_EPOCHS=8000
EMB_DIM=500
LOSS=Adv_Loss
TRAIN_BS=1024
EVAL_BS=4
NUM_NEG=400
MARGIN=200
LR=0.0002
REGULARIZATION=0.000002
CHECK_PER_EPOCH=500
NUM_WORKERS=4
GPU=0

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --negative_adversarial_sampling \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    --use_weight \
    --save_config