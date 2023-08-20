GPU_ID=6

DATASET='s3dis'
SPLIT=0
DATA_PATH='TFS3D-main/datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_s3dis_TFS3D/'

# DATASET='scannet'
# SPLIT=0
# DATA_PATH='TFS3D-main/datasets/ScanNet/blocks_bs1_s1'
# SAVE_PATH='./log_scannet_TFS3D/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
WAY_SAMP_RATIO='[0.05, 0.05]'
WAY_SAMP_NUM='[100, 100]'

N_WAY=2
K_SHOT=1
N_QUESIES=1
N_TEST_EPISODES=50

NUM_ITERS=24000
EVAL_INTERVAL=2000
LR=0.001
DECAY_STEP=7000
DECAY_RATIO=0.5


args=(--model 'training' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_augm_shift 0.1
      --way_pcratio "$WAY_SAMP_RATIO" --way_pcnum "$WAY_SAMP_NUM"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR  --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES
            )

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
