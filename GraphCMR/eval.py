#!/usr/bin/python
"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python eval.py --checkpoint=data/models/model_checkpoint_h36m_up3d.pt --config=data/config.json --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. UP-3D ```--dataset=up-3d```
4. LSP ```--dataset=lsp```
"""
from __future__ import print_function
from __future__ import division

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm

import config as cfg
from models import CMR, SMPL
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer
from utils.mesh import Mesh

from models.geometric_layers import orthographic_projection, rodrigues

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='up-3d', choices=['h36m-p1', 'h36m-p2', 'up-3d', 'lsp'], help='Choose evaluation dataset')
parser.add_argument('--config', default=None, help='Path to config file containing model architecture etc.')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')

def run_evaluation(model, dataset_name, dataset, 
                   mesh, batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """
    
    renderer = PartRenderer()
    
    # Create SMPL model
    smpl = SMPL().cuda()
    
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Transfer model to the GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    # 2D pose metrics
    pose_2d_m = np.zeros(len(dataset))
    pose_2d_m_smpl = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    eval_2d_pose = False
    eval_shape = False

    # Choose appropriate evaluation for each dataset
    if dataset_name == 'up-3d':
        eval_shape = True
    elif dataset_name == 'lsp':
        eval_2d_pose = True

    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl(gt_pose, gt_betas)
        images = batch['img'].to(torch.device('cuda'))
        curr_batch_size = images.shape[0]
        gt_keypoints_2d = batch['keypoints'].cpu().numpy()
        
        # Run inference
        with torch.no_grad():
            pred_vertices, pred_vertices_smpl, camera, pred_rotmat, pred_betas = model(images)

        # If mask or part evaluation, render the mask and part images
        if eval_2d_pose:
            mask, parts = renderer(pred_vertices, camera)

        # 2D pose evaluation (for LSP)
        if eval_2d_pose:
            for i in range(curr_batch_size):
                gt_kp = gt_keypoints_2d[i, list(range(14))]
                # Get 3D and projected 2D keypoints from the regressed shape
                pred_keypoints_3d = smpl.get_joints(pred_vertices)
                pred_keypoints_2d = orthographic_projection(pred_keypoints_3d, camera)[:, :, :2]
                pred_keypoints_3d_smpl = smpl.get_joints(pred_vertices_smpl)
                pred_keypoints_2d_smpl = orthographic_projection(pred_keypoints_3d_smpl, camera.detach())[:, :, :2]
                pred_kp = pred_keypoints_2d.cpu().numpy()[i, list(range(14))]
                pred_kp_smpl = pred_keypoints_2d_smpl.cpu().numpy()[i, list(range(14))]
                # Compute 2D pose losses
                loss_2d_pose = np.sum((gt_kp[: , :2] - pred_kp)**2)
                loss_2d_pose_smpl = np.sum((gt_kp[: , :2] - pred_kp_smpl)**2)

                #print(gt_kp)
                #print()
                #print(pred_kp_smpl)
                #raw_input("Press Enter to continue...")

                pose_2d_m[step * batch_size + i] = loss_2d_pose
                pose_2d_m_smpl[step * batch_size + i] = loss_2d_pose_smpl

        # Shape evaluation (Mean per-vertex error)
        if eval_shape:
            se = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            se_smpl = torch.sqrt(((pred_vertices_smpl - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            shape_err[step * batch_size:step * batch_size + curr_batch_size] = se
            shape_err_smpl[step * batch_size:step * batch_size + curr_batch_size] = se_smpl

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_2d_pose:
                print('2D keypoints (NonParam): ' + str(1000 * pose_2d_m[:step * batch_size].mean()))
                print('2D keypoints (Param): ' + str(1000 * pose_2d_m_smpl[:step * batch_size].mean()))
                print()
            if eval_shape:
                print('Shape Error (NonParam): ' + str(1000 * shape_err[:step * batch_size].mean()))
                print('Shape Error (Param): ' + str(1000 * shape_err_smpl[:step * batch_size].mean()))
                print()

    # Print and store final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_2d_pose:
        print('2D keypoints (NonParam): ' + str(1000 * pose_2d_m.mean()))
        print('2D keypoints (Param): ' + str(1000 * pose_2d_m_smpl.mean()))
        print()
        # store results
        #np.savez("../eval-output/CMR_no_extra_lsp_model_alt.npz", imgnames = dataset.imgname, kp_2d_err_graph = pose_2d_m, kp_2d_err_smpl = pose_2d_m_smpl)
    if eval_shape:
        print('Shape Error (NonParam): ' + str(1000 * shape_err.mean()))
        print('Shape Error (Param): ' + str(1000 * shape_err_smpl.mean()))
        print()
        # store results
        #np.savez("../eval-output/CMR_no_extra_up_3d_model_alt.npz", imgnames = dataset.imgname, shape_err_graph = shape_err, shape_err_smpl = shape_err_smpl)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)
    # Load model
    mesh = Mesh()
    model = CMR(mesh, options.num_layers, options.num_channels,
                      pretrained_checkpoint=args.checkpoint)
    # Setup evaluation dataset
    dataset = BaseDataset(options, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation(model, args.dataset, dataset, mesh,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)
