""" Util functions for loading and saving checkpoints
"""
import os
import torch

def load_model_checkpoint(model, model_checkpoint_path, optimizer=None, mode='test'):
    try:
        checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.tar'))
        start_iter = checkpoint['iteration']
        start_iou = checkpoint['IoU']
    except:
        raise ValueError('Model checkpoint file must be correctly given (%s).' %model_checkpoint_path)

    #print()
    # for k, v in checkpoint['model_state_dict'].items():
    #     #print(k, v.shape)
    #     print(k, v)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if mode == 'test':
        print('Load model checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return model
    else:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print('Checkpoint does not include optimizer state dict...')
        print('Resume from checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return model, optimizer

