# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from yacs.config import CfgNode as CN

_C = CN()

_C.agent = "our"
_C.tasks = "insert_onto_square_peg,open_drawer,place_wine_at_rack_location,light_bulb_in"
_C.exp_id = "def"
_C.resume = ""
# bs per device, effective bs is scaled by num device
_C.bs = 4
_C.epochs = 20
# number of dataloader workers, >= 0
_C.num_workers = 0
# 'transition_uniform' or 'task_uniform'
_C.sample_distribution_mode = 'transition_uniform'
_C.train_iter = 16 * 10000

# arguments present in both peract and rvt
# some of them donot support every possible combination in peract
_C.peract = CN()
_C.peract.lambda_weight_l2 = 1e-6
# lr should be thought on per sample basis
# effective lr is multiplied by bs * num_devices
_C.peract.lr = 2.5e-5
_C.peract.optimizer_type = "lamb"
_C.peract.warmup_steps = 0
_C.peract.lr_cos_dec = False
_C.peract.add_rgc_loss = True
_C.peract.num_rotation_classes = 72
_C.peract.amp = False
_C.peract.bnb = False
_C.peract.transform_augmentation = True
_C.peract.transform_augmentation_xyz = [0.1, 0.1, 0.1]
_C.peract.transform_augmentation_rpy = [0.0, 0.0, 20.0]

# arguments present in only rvt and not peract
_C.rvt = CN()
_C.rvt.gt_hm_sigma = 1.5
_C.rvt.img_aug = 0.1
_C.rvt.place_with_mean = True
_C.rvt.move_pc_in_bound = True

# CCF configuration
_C.ccf = CN()
_C.ccf.enabled = False

_C.ccf.num_train_candidates = 64
_C.ccf.num_infer_candidates = 128

_C.ccf.train_trans_sigma = 0.035
_C.ccf.infer_trans_sigma = 0.020
_C.ccf.train_rot_sigma_deg = 12.0
_C.ccf.infer_rot_sigma_deg = 8.0

_C.ccf.success_radius = 0.010
_C.ccf.success_rot_deg = 5.0
_C.ccf.contact_radius = 0.020
_C.ccf.collision_radius = 0.006

_C.ccf.max_pc_points = 4096

_C.ccf.success_loss_weight = 1.0
_C.ccf.contact_loss_weight = 0.5
_C.ccf.collision_loss_weight = 1.0
_C.ccf.residual_xyz_loss_weight = 1.0
_C.ccf.residual_rot_loss_weight = 0.25
_C.ccf.total_loss_weight = 0.2

_C.ccf.infer_contact_bonus = 0.25
_C.ccf.infer_collision_penalty = 1.0
_C.ccf.infer_residual_penalty = 0.05
_C.ccf.max_residual_xyz = 0.025
_C.ccf.max_residual_rot_deg = 5.0

_C.ccf.gripper_jaw_width = 0.080
_C.ccf.gripper_finger_length = 0.060
_C.ccf.gripper_finger_thickness = 0.012
_C.ccf.gripper_palm_depth = 0.020
_C.ccf.gripper_points_per_finger = 8
_C.ccf.gripper_points_on_palm = 8

_C.ccf.refine_position = True
_C.ccf.refine_rotation = True

# arguments present in peract official
_C.peract_official = CN()
_C.peract_official.cfg_path = "configs/peract_official_config.yaml"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
