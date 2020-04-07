"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

import config as cfg
from os.path import join
from src.torch_smpl import SMPL
import torch

def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main():
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    # Load evaluation data
    data = np.load(cfg.DATASET_FILES[0]['up-3d'])
    img_dir = cfg.DATASET_FOLDERS['up-3d']
    imgnames = data['imgname']
    all_gt_poses = data['pose'].astype(np.float)
    all_gt_shapes = data['shape'].astype(np.float)

    # Create SMPL model
    smpl = SMPL()

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(imgnames))

    # Store gt and estimated pose and shape parameters
    gt_poses_, gt_shapes_ = [], []
    est_poses_, est_shapes_ = [], []

    # Evaluation
    for i in range(len(imgnames)) :
        print(i)
        img_path = join(img_dir, str(imgnames[i]))
        json_path = None
        input_img, proc_param, img = preprocess_image(img_path, json_path)

        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
        # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)
        verts = torch.from_numpy(verts).float()
        
        est_pose = theta[0][3:75].reshape(1,72)
        est_shape = theta[0][75:85].reshape(1,10)
        est_pose = torch.from_numpy(est_pose).float()
        est_shape = torch.from_numpy(est_shape).float()
        est_verts = smpl(est_pose, est_shape)

        # Compute groundtruth vertices
        gt_pose = all_gt_poses[i].reshape(1,72)
        gt_pose = torch.from_numpy(gt_pose).float()
        gt_shape = all_gt_shapes[i].reshape(1,10)
        gt_shape = torch.from_numpy(gt_shape).float()
        gt_verts = smpl(gt_pose, gt_shape)

        print(gt_shape)
        print()
        print(est_shape)
        raw_input("Press Enter to continue...")

        # Shape evaluation (Mean per-vertex error)
        se = torch.sqrt(((verts - gt_verts) ** 2).sum(dim=-1)).mean(dim=-1).numpy()
        shape_err[i] = se[0]

        # Store gt and estimated pose and shape parameters
        gt_poses_.append(gt_pose.numpy())
        gt_shapes_.append(gt_shape.numpy())
        est_poses_.append(theta[0][3:75])
        est_shapes_.append(theta[0][75:85])

        #visualize(img, proc_param, joints[0], verts[0], cams[0])


    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    print('Shape Error : ' + str(1000 * shape_err.mean()))
    print()

    # Store final eval results
    #np.savez("../eval-output/HMR_up_3d.npz", imgnames = imgnames, shape_err_smpl = shape_err, gt_poses = gt_poses_, gt_shapes=gt_shapes_, est_poses=est_poses_, est_shapes=est_shapes_)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main()
