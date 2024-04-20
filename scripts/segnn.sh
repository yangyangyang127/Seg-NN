GPU_ID=0


####################  S3DIS dataset  #######################
DATASET='s3dis'
SPLIT=0    #  options: {0, 1}
DATA_PATH='datasets/S3DIS/blocks_bs1_s1'
NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
WAY_SAMP_RATIO='[0.15, 0.15]'
WAY_SAMP_NUM='[300, 300]'
SAVE_PATH='./log_s3dis_SegNN/'


###################  ScanNet dataset  ######################
# DATASET='scannet'
# SPLIT=0    #  options: {0, 1}
# DATA_PATH='datasets/ScanNet/blocks_bs1_s1'
# NUM_POINTS=2048
# PC_ATTRIBS='xyzrgbXYZ'
# WAY_SAMP_RATIO='[0.3, 0.3]'
# WAY_SAMP_NUM='[600, 600]'
# SAVE_PATH='./log_scannet_SegNN/'


N_WAY=2    #  options: {2, 3}
K_SHOT=1    #  options: {1, 5}
N_QUESIES=1
N_TEST_EPISODES=100


args=(--model 'segnn' --dataset "${DATASET}" --cvfold $SPLIT --save_path "$SAVE_PATH"
      --data_path  "$DATA_PATH" --way_pcratio "$WAY_SAMP_RATIO" --way_pcnum "$WAY_SAMP_NUM" --pc_augm_shift 0.1
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm --batch_size 1
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
