GPU_ID=1

# DATASET='s3dis'
# SPLIT=0
# DATA_PATH='TFS3D-main/datasets/S3DIS/blocks_bs1_s1'

DATASET='scannet'
SPLIT=1
DATA_PATH='TFS3D-main/datasets/ScanNet/blocks_bs1_s1'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
WAY_SAMP_RATIO='[0.1, 0.2]'
WAY_SAMP_NUM='[200, 400]'

N_WAY=2
K_SHOT=1
N_QUESIES=1
N_TEST_EPISODES=50

args=(--model 'training_free' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH"
      --way_pcratio "$WAY_SAMP_RATIO" --way_pcnum "$WAY_SAMP_NUM" --pc_augm_shift 0.1
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --batch_size 1
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES
            )

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"

