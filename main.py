"""Main function for this repo
"""
import ast
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument('--model', type=str, default='training', choices=['training_free', 'training'])
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options:{0,1}')
    parser.add_argument('--data_path', type=str, default='./datasets/S3DIS/blocks_bs1_s1',
                                                    help='Directory to the source data')
    parser.add_argument('--save_path', type=str, default='./log_s3dis/',
                        help='Directory to the save log and checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1500,
                        help='iteration/epoch inverval to evaluate model')

    #optimization
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers to load data')
    parser.add_argument('--n_iters', type=int, default=20000, help='number of iterations/epochs to train')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Model (eg. protoNet or MPTI) learning rate [default: 0.001]')
    parser.add_argument('--step_size', type=int, default=5000, help='Iterations of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

    #few-shot episode setting
    parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 1|3')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of samples/shots for each class: 1|5')
    parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
    parser.add_argument('--n_episode_test', type=int, default=100,
                        help='Number of episode per configuration during testing')

    # Point cloud processing
    parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
    parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')           
    parser.add_argument('--way_pcratio', default='[0.05, 0.05]', 
                        help='The least ratio of points for each target class in the point cloud.')
    parser.add_argument('--way_pcnum', default='[100, 100]', 
                        help='The least number of points for each target class in the point cloud.')
    parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
    parser.add_argument('--pc_augm_scale', type=float, default=0,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', type=int, default=1,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', type=int, default=1,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--pc_augm_shift', type=float, default=0,
                        help='Training augmentation: Probability of shifting points')
    parser.add_argument('--pc_augm_color', type=int, default=0,
                        help='Training augmentation: Bool, random color of all attributes')

    args = parser.parse_args()
    print(args)
    args.way_pcratio = ast.literal_eval(args.way_pcratio)
    args.way_pcnum = ast.literal_eval(args.way_pcnum)
    
    import datetime
    theTime = datetime.datetime.now()
    print(theTime)
    
    args.pc_in_dim = len(args.pc_attribs)

    # Start trainer for pre-train, proto-train, proto-eval, mpti-train, mpti-test
    if args.model == 'training':
        args.log_dir = args.save_path + 'log_%s_%s_S%d_N%d_K%d' % (args.dataset, args.model, args.cvfold, args.n_way, args.k_shot)
        from runs.training import train
        train(args)    
    elif args.model == 'training_free':
        args.log_dir = args.save_path + 'log_%s_%s_S%d_N%d_K%d' % (args.dataset, args.model, args.cvfold, args.n_way, args.k_shot)
        from runs.training_free import training_free
        training_free(args)
    else:
        raise ValueError('Please set correct phase.')
    
    print(theTime)
    theTime = datetime.datetime.now()
    print(theTime)
    
    