DATA_DIR=../datasets

MODEL_NAME=MyCompGCN
DATASET_NAME=FB15k-237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=MyCompGCNLitModel
TRAIN_SAMPLER_CLASS=CompGCNSampler
TEST_SAMPLER_CLASS=CompGCNTestSampler
MAX_EPOCHS=1000
EMB_DIM=128
LOSS_NAME=Cross_Entropy_Loss
TRAIN_BS=1024
EVAL_BS=256
NUM_NEG=1
LR=0.0005
NUM_BLOCKS=2
CHECK_PER_EPOCH=50
DECODER_MODEL=FNetE
OPN=mult
NUM_WORKERS=8
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -W ignore -u main.py  \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss_name $LOSS_NAME \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --lr $LR \
    --check_per_epoch $CHECK_PER_EPOCH \
    --decoder_model $DECODER_MODEL \
    --opn $OPN \
    --nblocks $NUM_BLOCKS \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    --save_config