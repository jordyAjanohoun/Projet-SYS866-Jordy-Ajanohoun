#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Eval output data file paths
    # Change here if necessary 
    OUTPUT_DATA_DIR = "eval-output/"
    CMR_EXTRA_2D_LSP_OUTPUT = "CMR_extra_2d_lsp.npz"
    CMR_EXTRA_2D_UP_3D_OUTPUT = "CMR_extra_2d_up_3d.npz"
    CMR_NO_EXTRA_LSP_OUTPUT = "CMR_no_extra_lsp.npz"
    CMR_NO_EXTRA_UP_3D_OUTPUT = "CMR_no_extra_up_3d.npz"
    HMR_LSP_OUTPUT = "HMR_lsp.npz"
    HMR_UP_3D = "HMR_up_3d.npz"


    # Load eval output data
    CMR_extra_2d_lsp = np.load(OUTPUT_DATA_DIR + CMR_EXTRA_2D_LSP_OUTPUT)
    CMR_extra_2d_up_3d = np.load(OUTPUT_DATA_DIR + CMR_EXTRA_2D_UP_3D_OUTPUT)
    CMR_no_extra_lsp = np.load(OUTPUT_DATA_DIR + CMR_NO_EXTRA_LSP_OUTPUT)
    CMR_no_extra_up_3d = np.load(OUTPUT_DATA_DIR + CMR_NO_EXTRA_UP_3D_OUTPUT)
    HMR_lsp = np.load(OUTPUT_DATA_DIR + HMR_LSP_OUTPUT)
    HMR_up_3d = np.load(OUTPUT_DATA_DIR + HMR_UP_3D)

    # Print content of loaded evaluation data
    print("")
    print("Content of loaded evaluation data : ")
    print("-- CMR_extra_2d_lsp : %s" % CMR_extra_2d_lsp.keys())
    print("-- CMR_extra_2d_up_3d : %s" % CMR_extra_2d_up_3d.keys())
    print("-- CMR_no_extra_lsp : %s" % CMR_no_extra_lsp.keys())
    print("-- CMR_no_extra_up_3d : %s" % CMR_no_extra_up_3d.keys())
    print("-- HMR_lsp : %s" % HMR_lsp.keys())
    print("-- HMR_up_3d : %s" % HMR_up_3d.keys())
    print("")

    # 3d error eval
    print("")
    print("Lsmpl loss (Mean per-vertex error) on up-3d dataset (x1000) : Mean | Std | Min | Max ")

    print("-- CMR_extra_2d_up_3d (Graph output mesh) : " 
    	+ str(1000 * CMR_extra_2d_up_3d['shape_err_graph'].mean()) + " | " 
    	+ str(np.std(CMR_extra_2d_up_3d['shape_err_graph'] * 1000)) + " | " 
    	+ str(np.amin(CMR_extra_2d_up_3d['shape_err_graph'] * 1000)) + " | "
    	+ str(np.amax(CMR_extra_2d_up_3d['shape_err_graph'] * 1000)) )

    print("-- CMR_extra_2d_up_3d (SMPL params output mesh) : " 
    	+ str(1000 * CMR_extra_2d_up_3d['shape_err_smpl'].mean()) + " | "
    	+ str(np.std(CMR_extra_2d_up_3d['shape_err_smpl'] * 1000)) + " | " 
    	+ str(np.amin(CMR_extra_2d_up_3d['shape_err_smpl'] * 1000)) + " | "
    	+ str(np.amax(CMR_extra_2d_up_3d['shape_err_smpl'] * 1000)) )

    print("-- CMR_no_extra_up_3d (Graph output mesh) : " 
    	+ str(1000 * CMR_no_extra_up_3d['shape_err_graph'].mean()) + " | "
        + str(np.std(CMR_no_extra_up_3d['shape_err_graph'] * 1000)) + " | " 
    	+ str(np.amin(CMR_no_extra_up_3d['shape_err_graph'] * 1000)) + " | "
    	+ str(np.amax(CMR_no_extra_up_3d['shape_err_graph'] * 1000)) )

    print("-- CMR_no_extra_up_3d (SMPL params output mesh) : " 
    	+ str(1000 * CMR_no_extra_up_3d['shape_err_smpl'].mean()) + " | "
        + str(np.std(CMR_no_extra_up_3d['shape_err_smpl'] * 1000)) + " | " 
    	+ str(np.amin(CMR_no_extra_up_3d['shape_err_smpl'] * 1000)) + " | "
    	+ str(np.amax(CMR_no_extra_up_3d['shape_err_smpl'] * 1000)) )

    print("-- HMR_up_3d : " 
    	+ str(1000 * HMR_up_3d['shape_err_smpl'].mean()) + " | "
    	+ str(np.std(HMR_up_3d['shape_err_smpl'] * 1000)) + " | " 
    	+ str(np.amin(HMR_up_3d['shape_err_smpl'] * 1000)) + " | "
    	+ str(np.amax(HMR_up_3d['shape_err_smpl'] * 1000)) )

    print("")
    
    # Groundtruth and estimated SMPL params for up-3d dataset using HMR
    all_gt_poses = HMR_up_3d['gt_poses'].reshape(1389,72)
    all_gt_shapes = HMR_up_3d['gt_shapes'].reshape(1389,10)
    all_est_poses = HMR_up_3d['est_poses']
    all_est_shapes = HMR_up_3d['est_shapes']
    # Compute SMPL parameters loss (MSE) with HMR on up-3d dataset
    MSE_pose_param = (np.square(all_gt_poses - all_est_poses)).mean(axis=1)
    MSE_shape_param = (np.square(all_gt_shapes - all_est_shapes)).mean(axis=1)

    # Print results
    print("SMPL parameters losses (MSE) with HMR on up-3d dataset : Mean | Std | Min | Max")

    print("-- HMR_up_3d (Pose param) : " 
    	+ str( MSE_pose_param.mean()) + " | "
    	+ str(np.std(MSE_pose_param)) + " | " 
    	+ str(np.amin(MSE_pose_param)) + " | "
    	+ str(np.amax(MSE_pose_param)) )

    print("-- HMR_up_3d (Shape param) : " 
    	+ str( MSE_shape_param.mean()) + " | "
    	+ str(np.std(MSE_shape_param)) + " | " 
    	+ str(np.amin(MSE_shape_param)) + " | "
    	+ str(np.amax(MSE_shape_param)) )

    print("")

    # 2d error eval
    print("2d joints reprojection error (MSE) on LSP dataset (x1000) : Mean | Std | Min | Max")

    print("-- CMR_extra_2d_lsp (Graph output mesh) : " 
    	+ str(1000 * CMR_extra_2d_lsp['kp_2d_err_graph'].mean()) + " | " 
    	+ str(np.std(CMR_extra_2d_lsp['kp_2d_err_graph'] * 1000)) + " | " 
    	+ str(np.amin(CMR_extra_2d_lsp['kp_2d_err_graph'] * 1000)) + " | "
    	+ str(np.amax(CMR_extra_2d_lsp['kp_2d_err_graph'] * 1000)) )

    print("-- CMR_extra_2d_lsp (SMPL params output mesh) : " 
    	+ str(1000 * CMR_extra_2d_lsp['kp_2d_err_smpl'].mean()) + " | "
    	+ str(np.std(CMR_extra_2d_lsp['kp_2d_err_smpl'] * 1000)) + " | " 
    	+ str(np.amin(CMR_extra_2d_lsp['kp_2d_err_smpl'] * 1000)) + " | "
    	+ str(np.amax(CMR_extra_2d_lsp['kp_2d_err_smpl'] * 1000)) )

    print("-- CMR_no_extra_lsp (Graph output mesh) : " 
    	+ str(1000 * CMR_no_extra_lsp['kp_2d_err_graph'].mean()) + " | " 
    	+ str(np.std(CMR_no_extra_lsp['kp_2d_err_graph'] * 1000)) + " | " 
    	+ str(np.amin(CMR_no_extra_lsp['kp_2d_err_graph'] * 1000)) + " | "
    	+ str(np.amax(CMR_no_extra_lsp['kp_2d_err_graph'] * 1000)) )

    print("-- CMR_no_extra_lsp (SMPL params output mesh) : " 
    	+ str(1000 * CMR_no_extra_lsp['kp_2d_err_smpl'].mean()) + " | "
    	+ str(np.std(CMR_no_extra_lsp['kp_2d_err_smpl'] * 1000)) + " | " 
    	+ str(np.amin(CMR_no_extra_lsp['kp_2d_err_smpl'] * 1000)) + " | "
    	+ str(np.amax(CMR_no_extra_lsp['kp_2d_err_smpl'] * 1000)) )

    print("-- HMR_lsp : " 
    	+ str(1000 * HMR_lsp['kp_2d_err_smpl'].mean()) + " | "
    	+ str(np.std(HMR_lsp['kp_2d_err_smpl'] * 1000)) + " | " 
    	+ str(np.amin(HMR_lsp['kp_2d_err_smpl'] * 1000)) + " | "
    	+ str(np.amax(HMR_lsp['kp_2d_err_smpl'] * 1000)) )

    print("")

    # Correlation between Mean per-vertex error and SMPL params error
    # Mean per-vertex error VS SMPL params error with HMR on UP-3D
    SMPL_params_loss = MSE_shape_param + MSE_pose_param
    plt.scatter(HMR_up_3d['shape_err_smpl'] * 1000, SMPL_params_loss/2)
    plt.title('Mean per-vertex error VS SMPL params error with HMR on UP-3D')
    plt.xlabel('Mean per-vertex error (x1000)')
    plt.ylabel('SMPL params error')
    plt.savefig("./mesh_VS_params_err_HMR_up-3d.png")

    plt.clf()

    # Target distance from mean shape VS mean per-vertex error on up-3d dataset
    plt.scatter(np.linalg.norm(all_gt_shapes,axis=1), CMR_extra_2d_up_3d['shape_err_graph'] * 10 , color='r', label='CMR_extra_2d_GRAPH')
    plt.scatter(np.linalg.norm(all_gt_shapes,axis=1), CMR_extra_2d_up_3d['shape_err_smpl'] * 10, color='g', label='CMR_extra_2d_SMPL')
    plt.scatter(np.linalg.norm(all_gt_shapes,axis=1), HMR_up_3d['shape_err_smpl'] * 10, color='b', label='HMR')
    plt.xlabel('Target distance from mean shape')
    plt.ylabel('Mean per-vertex error (x10)')
    plt.title('Distance from mean shape VS mean per-vertex error (x10) on up-3d')
    plt.legend(loc="upper right")
    plt.savefig("./distance_mean_shape_VS_accuracy.png")