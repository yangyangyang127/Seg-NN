""" Util functions for loading and saving checkpoints
"""
import os
import torch


def load_model_checkpoint(model_checkpoint_path):
    try:
        checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.pt'))
        iter = checkpoint['iteration']
        iou = checkpoint['IoU']
        print('Load model checkpoint at Iteration %d (IoU %f)...' % (iter, iou))
        return checkpoint['model']
    except:
        raise ValueError('Model checkpoint file must be correctly given (%s).' %model_checkpoint_path)
