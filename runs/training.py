""" Prototypical Network for Few-shot 3D Point Cloud Semantic Segmentation [Baseline]
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from runs.training_free import test_few_shot
from dataloaders.loader import MyDataset, MyTestDataset, batch_task_collate
from models.seg_learner import Learner
from utils.cuda_util import cast_cuda
from utils.logger import init_logger
from utils.checkpoint_util import load_model_checkpoint


def train(args):
    logger = init_logger(args.log_dir, args)
    PL = Learner(args)
    #Init datasets, dataloaders, and writer
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter,
                         'shift': args.pc_augm_shift,
                         'random_color': args.pc_augm_color,
                         }

    TRAIN_DATASET = MyDataset(args.data_path, args.dataset, cvfold=args.cvfold, num_episode=args.n_iters,
                              n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                              phase=args.model, mode='train',
                              num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                              pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG,
                              way_ratio=args.way_pcratio, way_num=args.way_pcnum)

    VALID_DATASET = MyTestDataset(args.model, args.data_path, args.dataset, cvfold=args.cvfold,
                                  num_episode_per_comb=args.n_episode_test,
                                  n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                  num_point=args.pc_npts, pc_attribs=args.pc_attribs, 
                                  way_ratio=args.way_pcratio, way_num=args.way_pcnum)
    
    VALID_CLASSES = list(VALID_DATASET.classes)

    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=1, collate_fn=batch_task_collate)
    VALID_LOADER = DataLoader(VALID_DATASET, batch_size=1, collate_fn=batch_task_collate)

    WRITER = SummaryWriter(log_dir=args.log_dir)
    
    TEST_DATASET = MyTestDataset(args.model, args.data_path, args.dataset, cvfold=args.cvfold,
                                 num_episode_per_comb=args.n_episode_test,
                                 n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                 num_point=args.pc_npts, pc_attribs=args.pc_attribs, 
                                 way_ratio=args.way_pcratio, way_num=args.way_pcnum, mode='test')
    
    TEST_CLASSES = list(TEST_DATASET.classes)
    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False, collate_fn=batch_task_collate)

    # train    
    best_iou = 0
    for batch_idx, (data, sampled_classes) in enumerate(TRAIN_LOADER):
        
        data = cast_cuda(data)
        loss, accuracy = PL.train(data)
        if (batch_idx+1) % 500 == 0:
            logger.cprint('=====[Train] Iter: %d | Loss: %.4f | Accuracy: %f =====' % (batch_idx, loss, accuracy))
            WRITER.add_scalar('Train/loss', loss, batch_idx)
            WRITER.add_scalar('Train/accuracy', accuracy, batch_idx)

        if (batch_idx+1) % args.eval_interval == 0:
            mean_IoU = test_few_shot(VALID_LOADER, PL, logger, VALID_CLASSES)
            logger.cprint('\n=====[Valid]  Mean IoU: %f =====\n' % (mean_IoU))

            WRITER.add_scalar('Valid/meanIoU', mean_IoU, batch_idx)
            if mean_IoU > best_iou:
                best_iou = mean_IoU
                logger.cprint('*******************Model Saved*******************')
                save_dict = {'iteration': batch_idx + 1,
                             'model': PL.model,
                             'IoU': best_iou}
                torch.save(save_dict, os.path.join(args.log_dir, 'checkpoint.pt'))

            logger.cprint('=====Mean Valid IoU Is: %f =====' % (mean_IoU))
            logger.cprint('=====Best Valid IoU Is: %f =====' % (best_iou))
                
    PL.model = load_model_checkpoint(args.log_dir)
    test_IoU = test_few_shot(TEST_LOADER, PL, logger, TEST_CLASSES)
    logger.cprint('\n=====[TEST]  Mean IoU: %f =====\n' % (test_IoU))            

    WRITER.close()
