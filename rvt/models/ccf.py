# Copyright (c) 2026.
# Counterfactual Contact Field 
# CCF module:
#   1. SE(3) counterfactual candidate sampling.
#   2. Gripper-proxy geometry for contact and collision supervision.
#   3. Contact, collision, success, translation residual, and rotation residual heads.
#   4. Inference-time pose refinement for both xyz and quaternion.

from typing import Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-8


def safe_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = _EPS,
) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.vector_norm(x, dim=dim, keepdim=True), min=eps)


def quat_normalize(
    quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    return safe_normalize(quat_xyzw, dim=-1)


def quat_conjugate(
    quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    out = quat_xyzw.clone()
    out[..., 0:3] = -out[..., 0:3]
    return out


def quat_mul(
    quat_a_xyzw: torch.Tensor,
    quat_b_xyzw: torch.Tensor,
) -> torch.Tensor:
    ax = quat_a_xyzw[..., 0]
    ay = quat_a_xyzw[..., 1]
    az = quat_a_xyzw[..., 2]
    aw = quat_a_xyzw[..., 3]

    bx = quat_b_xyzw[..., 0]
    by = quat_b_xyzw[..., 1]
    bz = quat_b_xyzw[..., 2]
    bw = quat_b_xyzw[..., 3]

    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz

    return quat_normalize(torch.stack([x, y, z, w], dim=-1))


def axis_angle_to_quat(
    axis_angle: torch.Tensor,
) -> torch.Tensor:
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / torch.clamp(angle, min=_EPS)
    half_angle = 0.5 * angle
    sin_half = torch.sin(half_angle)
    xyz = axis * sin_half
    w = torch.cos(half_angle)
    quat = torch.cat([xyz, w], dim=-1)
    small = angle < 1e-7
    identity = torch.zeros_like(quat)
    identity[..., 3] = 1.0
    quat = torch.where(small.expand_as(quat), identity, quat)
    return quat_normalize(quat)


def quat_to_axis_angle(
    quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    quat_xyzw = quat_normalize(quat_xyzw)
    xyz = quat_xyzw[..., 0:3]
    w = torch.clamp(quat_xyzw[..., 3:4], min=-1.0, max=1.0)
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    axis = xyz / torch.clamp(sin_half, min=_EPS)
    axis_angle = axis * angle
    axis_angle = torch.where(
        sin_half.expand_as(axis_angle) < 1e-7,
        torch.zeros_like(axis_angle),
        axis_angle,
    )
    return axis_angle


def quat_rotate(
    quat_xyzw: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    quat_xyzw = quat_normalize(quat_xyzw)

    q_xyz = quat_xyzw[..., 0:3]
    q_w = quat_xyzw[..., 3:4]

    while q_xyz.ndim < points.ndim:
        q_xyz = q_xyz.unsqueeze(-2)
        q_w = q_w.unsqueeze(-2)

    t = 2.0 * torch.cross(q_xyz.expand_as(points), points, dim=-1)
    return points + q_w.expand_as(points) * t + torch.cross(q_xyz.expand_as(points), t, dim=-1)


def quat_geodesic_distance(
    quat_a_xyzw: torch.Tensor,
    quat_b_xyzw: torch.Tensor,
) -> torch.Tensor:
    quat_a_xyzw = quat_normalize(quat_a_xyzw)
    quat_b_xyzw = quat_normalize(quat_b_xyzw)
    dot = torch.sum(quat_a_xyzw * quat_b_xyzw, dim=-1).abs()
    dot = torch.clamp(dot, min=-1.0, max=1.0)
    return 2.0 * torch.acos(dot)


def make_gripper_proxy_points(
    device: torch.device,
    dtype: torch.dtype,
    jaw_width: float,
    finger_length: float,
    finger_thickness: float,
    palm_depth: float,
    points_per_finger: int,
    points_on_palm: int,
) -> torch.Tensor:
    xs = torch.linspace(
        -finger_length * 0.5,
        finger_length * 0.5,
        points_per_finger,
        device=device,
        dtype=dtype,
    )

    y_left = torch.full_like(xs, jaw_width * 0.5)
    y_right = torch.full_like(xs, -jaw_width * 0.5)
    z_mid = torch.zeros_like(xs)

    left = torch.stack([xs, y_left, z_mid], dim=-1)
    right = torch.stack([xs, y_right, z_mid], dim=-1)

    ys = torch.linspace(
        -jaw_width * 0.5,
        jaw_width * 0.5,
        points_on_palm,
        device=device,
        dtype=dtype,
    )
    x_palm = torch.full_like(ys, -finger_length * 0.5 - palm_depth)
    z_palm = torch.zeros_like(ys)
    palm = torch.stack([x_palm, ys, z_palm], dim=-1)

    top_offset = torch.tensor(
        [0.0, 0.0, finger_thickness * 0.5],
        device=device,
        dtype=dtype,
    )
    bottom_offset = torch.tensor(
        [0.0, 0.0, -finger_thickness * 0.5],
        device=device,
        dtype=dtype,
    )

    proxy = torch.cat(
        [
            left + top_offset,
            left + bottom_offset,
            right + top_offset,
            right + bottom_offset,
            palm + top_offset,
            palm + bottom_offset,
        ],
        dim=0,
    )

    return proxy


def transform_gripper_proxy(
    candidate_xyz: torch.Tensor,
    candidate_quat_xyzw: torch.Tensor,
    proxy_points: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_candidates, _ = candidate_xyz.shape
    num_proxy_points = proxy_points.shape[0]

    proxy = proxy_points.view(1, 1, num_proxy_points, 3)
    proxy = proxy.expand(batch_size, num_candidates, num_proxy_points, 3)

    quat = candidate_quat_xyzw.unsqueeze(2).expand(
        batch_size,
        num_candidates,
        num_proxy_points,
        4,
    )

    rotated = quat_rotate(quat, proxy)
    world = rotated + candidate_xyz.unsqueeze(2)
    return world


def subsample_pc(
    pc: torch.Tensor,
    max_points: int,
) -> torch.Tensor:
    if pc.shape[0] <= max_points:
        return pc
    index = torch.randperm(pc.shape[0], device=pc.device)[:max_points]
    return pc[index]


def nearest_distance_to_scene(
    query_points: torch.Tensor,
    pc_list: List[torch.Tensor],
    max_points: int,
) -> torch.Tensor:
    batch_size = query_points.shape[0]
    flat_shape = query_points.shape[:-1]
    query_flat = query_points.reshape(batch_size, -1, 3)

    all_dist = []

    for batch_idx in range(batch_size):
        pc = pc_list[batch_idx]

        if pc.ndim != 2 or pc.shape[-1] != 3:
            raise ValueError(f"Expected each point cloud to have shape (N, 3), got {pc.shape}.")

        if pc.shape[0] == 0:
            dist = torch.full(
                (query_flat.shape[1],),
                fill_value=1e6,
                device=query_points.device,
                dtype=query_points.dtype,
            )
            all_dist.append(dist)
            continue

        pc = subsample_pc(pc, max_points=max_points)
        dist = torch.cdist(
            query_flat[batch_idx].unsqueeze(0),
            pc.unsqueeze(0),
            p=2,
        ).squeeze(0)
        min_dist = dist.min(dim=-1).values
        all_dist.append(min_dist)

    all_dist = torch.stack(all_dist, dim=0)
    return all_dist.reshape(*flat_shape)


class CounterfactualContactField(nn.Module):
    def __init__(
        self,
        scene_feat_dim: int,
        pose_feat_dim: int = 17,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2.")

        self.scene_feat_dim = int(scene_feat_dim)
        self.pose_feat_dim = int(pose_feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        layers = []
        input_dim = self.scene_feat_dim + self.pose_feat_dim

        for layer_idx in range(self.num_layers - 1):
            in_dim = input_dim if layer_idx == 0 else self.hidden_dim
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.GELU())

            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))

        self.trunk = nn.Sequential(*layers)
        self.success_head = nn.Linear(self.hidden_dim, 1)
        self.contact_head = nn.Linear(self.hidden_dim, 1)
        self.collision_head = nn.Linear(self.hidden_dim, 1)
        self.delta_xyz_head = nn.Linear(self.hidden_dim, 3)
        self.delta_rot_axis_angle_head = nn.Linear(self.hidden_dim, 3)

    def forward(
        self,
        scene_feat: torch.Tensor,
        cand_pose_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if scene_feat.ndim != 2:
            raise ValueError(f"scene_feat must have shape (B, F), got {scene_feat.shape}.")

        if cand_pose_feat.ndim != 3:
            raise ValueError(f"cand_pose_feat must have shape (B, K, D), got {cand_pose_feat.shape}.")

        batch_size, num_candidates, pose_feat_dim = cand_pose_feat.shape

        if pose_feat_dim != self.pose_feat_dim:
            raise ValueError(
                f"Expected cand_pose_feat last dim {self.pose_feat_dim}, got {pose_feat_dim}."
            )

        if scene_feat.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch: scene_feat has {scene_feat.shape[0]}, candidates have {batch_size}."
            )

        scene_feat = scene_feat.unsqueeze(1).expand(-1, num_candidates, -1)
        fused = torch.cat([scene_feat, cand_pose_feat], dim=-1)
        fused = fused.reshape(batch_size * num_candidates, -1)

        hidden = self.trunk(fused)

        success_logit = self.success_head(hidden).reshape(batch_size, num_candidates)
        contact_logit = self.contact_head(hidden).reshape(batch_size, num_candidates)
        collision_logit = self.collision_head(hidden).reshape(batch_size, num_candidates)
        delta_xyz = self.delta_xyz_head(hidden).reshape(batch_size, num_candidates, 3)
        delta_rot_axis_angle = self.delta_rot_axis_angle_head(hidden).reshape(
            batch_size,
            num_candidates,
            3,
        )

        return {
            "success_logit": success_logit,
            "contact_logit": contact_logit,
            "collision_logit": collision_logit,
            "delta_xyz": delta_xyz,
            "delta_rot_axis_angle": delta_rot_axis_angle,
        }


def get_ccf_scene_feat(
    out: Dict[str, torch.Tensor],
    stage_two: bool,
) -> torch.Tensor:
    if stage_two:
        if "mvt2" not in out:
            raise KeyError("stage_two=True, but MVT output does not contain 'mvt2'.")
        out = out["mvt2"]

    if "feat_pre" in out:
        return out["feat_pre"]

    if "feat_ex_rot" in out:
        return out["feat_ex_rot"].reshape(out["feat_ex_rot"].shape[0], -1)

    if "feat" in out:
        return out["feat"].reshape(out["feat"].shape[0], -1)

    raise KeyError("Cannot find feature for CCF. Expected feat_pre, feat_ex_rot, or feat.")


def build_pose_feature(
    candidate_xyz: torch.Tensor,
    candidate_quat_xyzw: torch.Tensor,
    center_xyz: torch.Tensor,
    center_quat_xyzw: torch.Tensor,
    trans_sigma: float,
    rot_sigma_deg: float,
) -> torch.Tensor:
    delta_xyz = candidate_xyz - center_xyz.unsqueeze(1)

    safe_trans_sigma = max(float(trans_sigma), 1e-6)
    delta_xyz_norm = delta_xyz / safe_trans_sigma

    center_quat = center_quat_xyzw.unsqueeze(1).expand_as(candidate_quat_xyzw)
    delta_quat = quat_mul(candidate_quat_xyzw, quat_conjugate(center_quat))
    delta_axis_angle = quat_to_axis_angle(delta_quat)

    safe_rot_sigma = max(math.radians(float(rot_sigma_deg)), 1e-6)
    delta_axis_angle_norm = delta_axis_angle / safe_rot_sigma

    delta_xyz_radius = torch.linalg.vector_norm(
        delta_xyz_norm,
        dim=-1,
        keepdim=True,
    )

    pose_feat = torch.cat(
        [
            delta_xyz_norm,
            delta_axis_angle_norm,
            delta_xyz_radius,
            candidate_xyz,
            candidate_quat_xyzw,
            center_xyz.unsqueeze(1).expand_as(candidate_xyz),
        ],
        dim=-1,
    )

    return pose_feat


def sample_counterfactual_se3_candidates(
    center_xyz: torch.Tensor,
    center_quat_xyzw: torch.Tensor,
    num_candidates: int,
    trans_sigma: float,
    rot_sigma_deg: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if center_xyz.ndim != 2 or center_xyz.shape[-1] != 3:
        raise ValueError(f"center_xyz must have shape (B, 3), got {center_xyz.shape}.")

    if center_quat_xyzw.ndim != 2 or center_quat_xyzw.shape[-1] != 4:
        raise ValueError(f"center_quat_xyzw must have shape (B, 4), got {center_quat_xyzw.shape}.")

    if num_candidates < 2:
        raise ValueError("num_candidates must be >= 2.")

    batch_size = center_xyz.shape[0]
    device = center_xyz.device
    dtype = center_xyz.dtype

    delta_xyz = torch.randn(
        batch_size,
        num_candidates,
        3,
        device=device,
        dtype=dtype,
    ) * float(trans_sigma)
    delta_xyz[:, 0, :] = 0.0

    rot_sigma_rad = math.radians(float(rot_sigma_deg))
    delta_axis_angle = torch.randn(
        batch_size,
        num_candidates,
        3,
        device=device,
        dtype=dtype,
    ) * rot_sigma_rad
    delta_axis_angle[:, 0, :] = 0.0

    delta_quat = axis_angle_to_quat(delta_axis_angle)
    base_quat = center_quat_xyzw.unsqueeze(1).expand_as(delta_quat)

    candidate_xyz = center_xyz.unsqueeze(1) + delta_xyz
    candidate_quat = quat_mul(delta_quat, base_quat)

    return candidate_xyz, candidate_quat


def build_ccf_training_batch(
    wpt_local: torch.Tensor,
    action_rot_quat: torch.Tensor,
    pc_list: List[torch.Tensor],
    num_candidates: int,
    trans_sigma: float,
    rot_sigma_deg: float,
    success_radius: float,
    success_rot_deg: float,
    contact_radius: float,
    collision_radius: float,
    max_pc_points: int,
    gripper_jaw_width: float,
    gripper_finger_length: float,
    gripper_finger_thickness: float,
    gripper_palm_depth: float,
    gripper_points_per_finger: int,
    gripper_points_on_palm: int,
) -> Dict[str, torch.Tensor]:
    if wpt_local.ndim != 2 or wpt_local.shape[-1] != 3:
        raise ValueError(f"wpt_local must have shape (B, 3), got {wpt_local.shape}.")

    if action_rot_quat.ndim != 2 or action_rot_quat.shape[-1] != 4:
        raise ValueError(f"action_rot_quat must have shape (B, 4), got {action_rot_quat.shape}.")

    action_rot_quat = quat_normalize(action_rot_quat)

    candidate_xyz, candidate_quat = sample_counterfactual_se3_candidates(
        center_xyz=wpt_local,
        center_quat_xyzw=action_rot_quat,
        num_candidates=num_candidates,
        trans_sigma=trans_sigma,
        rot_sigma_deg=rot_sigma_deg,
    )

    pose_feat = build_pose_feature(
        candidate_xyz=candidate_xyz,
        candidate_quat_xyzw=candidate_quat,
        center_xyz=wpt_local,
        center_quat_xyzw=action_rot_quat,
        trans_sigma=trans_sigma,
        rot_sigma_deg=rot_sigma_deg,
    )

    trans_error = torch.linalg.vector_norm(
        candidate_xyz - wpt_local.unsqueeze(1),
        dim=-1,
    )

    rot_error = quat_geodesic_distance(
        candidate_quat,
        action_rot_quat.unsqueeze(1).expand_as(candidate_quat),
    )

    success_target = (
        (trans_error <= float(success_radius))
        & (rot_error <= math.radians(float(success_rot_deg)))
    ).float()

    proxy_points = make_gripper_proxy_points(
        device=wpt_local.device,
        dtype=wpt_local.dtype,
        jaw_width=float(gripper_jaw_width),
        finger_length=float(gripper_finger_length),
        finger_thickness=float(gripper_finger_thickness),
        palm_depth=float(gripper_palm_depth),
        points_per_finger=int(gripper_points_per_finger),
        points_on_palm=int(gripper_points_on_palm),
    )

    gripper_points = transform_gripper_proxy(
        candidate_xyz=candidate_xyz,
        candidate_quat_xyzw=candidate_quat,
        proxy_points=proxy_points,
    )

    proxy_dist = nearest_distance_to_scene(
        query_points=gripper_points,
        pc_list=pc_list,
        max_points=int(max_pc_points),
    )

    min_proxy_dist = proxy_dist.min(dim=-1).values
    mean_proxy_dist = proxy_dist.mean(dim=-1)

    collision_target = (min_proxy_dist <= float(collision_radius)).float()

    contact_target = (
        (min_proxy_dist > float(collision_radius))
        & (min_proxy_dist <= float(contact_radius))
    ).float()

    residual_xyz_target = wpt_local.unsqueeze(1) - candidate_xyz

    residual_rot_quat = quat_mul(
        action_rot_quat.unsqueeze(1).expand_as(candidate_quat),
        quat_conjugate(candidate_quat),
    )
    residual_rot_axis_angle_target = quat_to_axis_angle(residual_rot_quat)

    return {
        "cand_pose_feat": pose_feat,
        "candidate_xyz": candidate_xyz,
        "candidate_quat": candidate_quat,
        "success_target": success_target,
        "contact_target": contact_target,
        "collision_target": collision_target,
        "residual_xyz_target": residual_xyz_target,
        "residual_rot_axis_angle_target": residual_rot_axis_angle_target,
        "trans_error": trans_error,
        "rot_error": rot_error,
        "min_proxy_dist": min_proxy_dist,
        "mean_proxy_dist": mean_proxy_dist,
    }


def build_ccf_inference_batch(
    center_xyz: torch.Tensor,
    center_quat_xyzw: torch.Tensor,
    num_candidates: int,
    trans_sigma: float,
    rot_sigma_deg: float,
) -> Dict[str, torch.Tensor]:
    candidate_xyz, candidate_quat = sample_counterfactual_se3_candidates(
        center_xyz=center_xyz,
        center_quat_xyzw=center_quat_xyzw,
        num_candidates=num_candidates,
        trans_sigma=trans_sigma,
        rot_sigma_deg=rot_sigma_deg,
    )

    pose_feat = build_pose_feature(
        candidate_xyz=candidate_xyz,
        candidate_quat_xyzw=candidate_quat,
        center_xyz=center_xyz,
        center_quat_xyzw=center_quat_xyzw,
        trans_sigma=trans_sigma,
        rot_sigma_deg=rot_sigma_deg,
    )

    return {
        "cand_pose_feat": pose_feat,
        "candidate_xyz": candidate_xyz,
        "candidate_quat": candidate_quat,
    }


def compute_ccf_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    success_weight: float,
    contact_weight: float,
    collision_weight: float,
    residual_xyz_weight: float,
    residual_rot_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    success_loss = F.binary_cross_entropy_with_logits(
        pred["success_logit"],
        target["success_target"],
    )

    contact_loss = F.binary_cross_entropy_with_logits(
        pred["contact_logit"],
        target["contact_target"],
    )

    collision_loss = F.binary_cross_entropy_with_logits(
        pred["collision_logit"],
        target["collision_target"],
    )

    residual_xyz_loss = F.smooth_l1_loss(
        pred["delta_xyz"],
        target["residual_xyz_target"],
    )

    residual_rot_loss = F.smooth_l1_loss(
        pred["delta_rot_axis_angle"],
        target["residual_rot_axis_angle_target"],
    )

    total_loss = (
        float(success_weight) * success_loss
        + float(contact_weight) * contact_loss
        + float(collision_weight) * collision_loss
        + float(residual_xyz_weight) * residual_xyz_loss
        + float(residual_rot_weight) * residual_rot_loss
    )

    with torch.no_grad():
        success_pred = (torch.sigmoid(pred["success_logit"]) >= 0.5).float()
        contact_pred = (torch.sigmoid(pred["contact_logit"]) >= 0.5).float()
        collision_pred = (torch.sigmoid(pred["collision_logit"]) >= 0.5).float()

        success_acc = (success_pred == target["success_target"]).float().mean()
        contact_acc = (contact_pred == target["contact_target"]).float().mean()
        collision_acc = (collision_pred == target["collision_target"]).float().mean()

    log = {
        "ccf_loss": float(total_loss.detach().cpu()),
        "ccf_success_loss": float(success_loss.detach().cpu()),
        "ccf_contact_loss": float(contact_loss.detach().cpu()),
        "ccf_collision_loss": float(collision_loss.detach().cpu()),
        "ccf_residual_xyz_loss": float(residual_xyz_loss.detach().cpu()),
        "ccf_residual_rot_loss": float(residual_rot_loss.detach().cpu()),
        "ccf_success_acc": float(success_acc.detach().cpu()),
        "ccf_contact_acc": float(contact_acc.detach().cpu()),
        "ccf_collision_acc": float(collision_acc.detach().cpu()),
        "ccf_positive_success_ratio": float(target["success_target"].mean().detach().cpu()),
        "ccf_positive_contact_ratio": float(target["contact_target"].mean().detach().cpu()),
        "ccf_positive_collision_ratio": float(target["collision_target"].mean().detach().cpu()),
        "ccf_min_proxy_dist": float(target["min_proxy_dist"].mean().detach().cpu()),
        "ccf_mean_proxy_dist": float(target["mean_proxy_dist"].mean().detach().cpu()),
    }

    return total_loss, log


@torch.no_grad()
def refine_pose_with_ccf(
    ccf_head: CounterfactualContactField,
    scene_feat: torch.Tensor,
    center_xyz: torch.Tensor,
    center_quat_xyzw: torch.Tensor,
    num_candidates: int,
    trans_sigma: float,
    rot_sigma_deg: float,
    contact_bonus: float,
    collision_penalty: float,
    residual_penalty: float,
    max_residual_xyz: float,
    max_residual_rot_deg: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    infer_batch = build_ccf_inference_batch(
        center_xyz=center_xyz,
        center_quat_xyzw=center_quat_xyzw,
        num_candidates=num_candidates,
        trans_sigma=trans_sigma,
        rot_sigma_deg=rot_sigma_deg,
    )

    pred = ccf_head(
        scene_feat=scene_feat,
        cand_pose_feat=infer_batch["cand_pose_feat"],
    )

    success_score = torch.sigmoid(pred["success_logit"])
    contact_score = torch.sigmoid(pred["contact_logit"])
    collision_score = torch.sigmoid(pred["collision_logit"])

    residual_xyz_norm = torch.linalg.vector_norm(pred["delta_xyz"], dim=-1)
    residual_rot_norm = torch.linalg.vector_norm(pred["delta_rot_axis_angle"], dim=-1)

    score = (
        success_score
        + float(contact_bonus) * contact_score
        - float(collision_penalty) * collision_score
        - float(residual_penalty) * (residual_xyz_norm + residual_rot_norm)
    )

    best_idx = score.argmax(dim=1)
    batch_idx = torch.arange(center_xyz.shape[0], device=center_xyz.device)

    best_xyz = infer_batch["candidate_xyz"][batch_idx, best_idx]
    best_quat = infer_batch["candidate_quat"][batch_idx, best_idx]

    delta_xyz = pred["delta_xyz"][batch_idx, best_idx]
    delta_xyz = torch.clamp(
        delta_xyz,
        min=-float(max_residual_xyz),
        max=float(max_residual_xyz),
    )

    delta_rot_axis_angle = pred["delta_rot_axis_angle"][batch_idx, best_idx]
    max_rot = math.radians(float(max_residual_rot_deg))
    delta_rot_axis_angle = torch.clamp(
        delta_rot_axis_angle,
        min=-max_rot,
        max=max_rot,
    )

    delta_quat = axis_angle_to_quat(delta_rot_axis_angle)

    refined_xyz = best_xyz + delta_xyz
    refined_quat = quat_mul(delta_quat, best_quat)

    info = {
        "ccf_best_idx": best_idx,
        "ccf_best_score": score[batch_idx, best_idx],
        "ccf_success_score": success_score[batch_idx, best_idx],
        "ccf_contact_score": contact_score[batch_idx, best_idx],
        "ccf_collision_score": collision_score[batch_idx, best_idx],
        "ccf_delta_xyz_norm": torch.linalg.vector_norm(delta_xyz, dim=-1),
        "ccf_delta_rot_norm": torch.linalg.vector_norm(delta_rot_axis_angle, dim=-1),
    }

    return refined_xyz, refined_quat, info