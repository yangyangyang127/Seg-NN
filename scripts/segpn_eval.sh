GPU_ID=0

####################  S3DIS dataset  #######################
# DATASET='s3dis'
# SPLIT=0    #  options: {0, 1}
# DATA_PATH='datasets/S3DIS/blocks_bs1_s1'
# SAVE_PATH='./log_s3dis_SegPN/'


###################  ScanNet dataset  ######################
DATASET='scannet'
SPLIT=0    #  options: {0, 1}
DATA_PATH='datasets/ScanNet/blocks_bs1_s1'
SAVE_PATH='./log_scannet_SegPN/'


NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
WAY_SAMP_RATIO='[0.05, 0.05]'
WAY_SAMP_NUM='[100, 100]'

N_WAY=2    #  options: {2, 3}
K_SHOT=1    #  options: {1, 5}
N_QUESIES=1
N_TEST_EPISODES=100

args=(--model 'segpn_eval' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --way_pcratio "$WAY_SAMP_RATIO" --way_pcnum "$WAY_SAMP_NUM"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --batch_size 1 --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
