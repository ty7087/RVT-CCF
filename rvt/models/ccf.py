# Copyright (c) 2026.
# This file adds Counterfactual Contact Field support on top of RVT-2.
#
# The code in this file is intentionally independent from the official RVT-2
# implementation, so it can be added without changing renderer or RLBench code.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualContactField(nn.Module):
    """
    Counterfactual Contact Field.

    The module receives:
        scene_feat:
            Tensor with shape (B, F). It is extracted from the fine-stage MVT.
        cand_pose_feat:
            Tensor with shape (B, K, 9). It describes K local candidate poses
            around the current waypoint.

    The module predicts:
        success_logit:
            Tensor with shape (B, K). Higher means the candidate is likely
            to be a physically useful contact pose.
        collision_logit:
            Tensor with shape (B, K). Higher means the candidate is likely
            to collide or be geometrically unsafe.
        delta_xyz:
            Tensor with shape (B, K, 3). A local residual correction from
            candidate position to a better position.
    """

    def __init__(
        self,
        scene_feat_dim: int,
        pose_feat_dim: int = 9,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        self.scene_feat_dim = int(scene_feat_dim)
        self.pose_feat_dim = int(pose_feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        layers = []
        in_dim = self.scene_feat_dim + self.pose_feat_dim

        for layer_id in range(self.num_layers - 1):
            layers.append(nn.Linear(in_dim if layer_id == 0 else self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.GELU())

            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))

        self.trunk = nn.Sequential(*layers)
        self.success_head = nn.Linear(self.hidden_dim, 1)
        self.collision_head = nn.Linear(self.hidden_dim, 1)
        self.delta_head = nn.Linear(self.hidden_dim, 3)

    def forward(
        self,
        scene_feat: torch.Tensor,
        cand_pose_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if scene_feat.ndim != 2:
            raise ValueError(f"scene_feat must have shape (B, F), got {scene_feat.shape}.")

        if cand_pose_feat.ndim != 3:
            raise ValueError(
                f"cand_pose_feat must have shape (B, K, D), got {cand_pose_feat.shape}."
            )

        batch_size, num_candidates, pose_dim = cand_pose_feat.shape

        if scene_feat.shape[0] != batch_size:
            raise ValueError(
                "scene_feat and cand_pose_feat must have the same batch size: "
                f"{scene_feat.shape[0]} vs {batch_size}."
            )

        if scene_feat.shape[1] != self.scene_feat_dim:
            raise ValueError(
                f"Expected scene_feat dim {self.scene_feat_dim}, got {scene_feat.shape[1]}."
            )

        if pose_dim != self.pose_feat_dim:
            raise ValueError(f"Expected pose feature dim {self.pose_feat_dim}, got {pose_dim}.")

        scene_feat = scene_feat.unsqueeze(1).expand(-1, num_candidates, -1)
        fused = torch.cat([scene_feat, cand_pose_feat], dim=-1)
        fused = fused.reshape(batch_size * num_candidates, -1)

        hidden = self.trunk(fused)

        success_logit = self.success_head(hidden).reshape(batch_size, num_candidates)
        collision_logit = self.collision_head(hidden).reshape(batch_size, num_candidates)
        delta_xyz = self.delta_head(hidden).reshape(batch_size, num_candidates, 3)

        return {
            "success_logit": success_logit,
            "collision_logit": collision_logit,
            "delta_xyz": delta_xyz,
        }


def get_ccf_scene_feat(
    out: Dict[str, torch.Tensor],
    stage_two: bool,
) -> torch.Tensor:
    """
    Extract a scene-level feature tensor from MVT output.

    In RVT-2, the final action heads are produced by the second-stage MVT,
    so we prefer out["mvt2"]["feat_pre"] when stage_two=True.
    """

    if stage_two:
        if "mvt2" not in out:
            raise KeyError("stage_two=True, but out does not contain key 'mvt2'.")
        out = out["mvt2"]

    if "feat_pre" in out:
        return out["feat_pre"]

    if "feat_ex_rot" in out:
        return out["feat_ex_rot"].reshape(out["feat_ex_rot"].shape[0], -1)

    if "feat" in out:
        return out["feat"].reshape(out["feat"].shape[0], -1)

    raise KeyError(
        "Cannot extract CCF scene feature. Expected one of: "
        "'feat_pre', 'feat_ex_rot', or 'feat'."
    )


def _subsample_pc(
    pc: torch.Tensor,
    max_points: int,
) -> torch.Tensor:
    if pc.shape[0] <= max_points:
        return pc

    rand_idx = torch.randperm(pc.shape[0], device=pc.device)[:max_points]
    return pc[rand_idx]


def nearest_pc_distance(
    candidates_xyz: torch.Tensor,
    pc_list: List[torch.Tensor],
    max_points: int = 4096,
) -> torch.Tensor:
    """
    Compute nearest point-cloud distance for each candidate.

    Args:
        candidates_xyz:
            Tensor with shape (B, K, 3).
        pc_list:
            List of B tensors. Each tensor has shape (N_i, 3).
        max_points:
            Point-cloud subsampling limit.

    Returns:
        Tensor with shape (B, K).
    """

    if candidates_xyz.ndim != 3:
        raise ValueError(f"candidates_xyz must have shape (B, K, 3), got {candidates_xyz.shape}.")

    batch_size, num_candidates, xyz_dim = candidates_xyz.shape

    if xyz_dim != 3:
        raise ValueError(f"Expected xyz dimension 3, got {xyz_dim}.")

    if len(pc_list) != batch_size:
        raise ValueError(f"pc_list length {len(pc_list)} does not match batch size {batch_size}.")

    all_dist = []

    for batch_idx in range(batch_size):
        pc = pc_list[batch_idx]

        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"Each point cloud must have shape (N, 3), got {pc.shape}.")

        if pc.shape[0] == 0:
            inf_dist = torch.full(
                (num_candidates,),
                fill_value=1e6,
                device=candidates_xyz.device,
                dtype=candidates_xyz.dtype,
            )
            all_dist.append(inf_dist)
            continue

        pc = _subsample_pc(pc, max_points=max_points)
        cand = candidates_xyz[batch_idx]

        dist = torch.cdist(cand.unsqueeze(0), pc.unsqueeze(0), p=2).squeeze(0)
        min_dist = dist.min(dim=1).values
        all_dist.append(min_dist)

    return torch.stack(all_dist, dim=0)


def build_ccf_training_batch(
    wpt_local: torch.Tensor,
    pc_list: List[torch.Tensor],
    num_candidates: int,
    trans_sigma: float,
    success_radius: float,
    collision_radius: float,
    max_points: int,
) -> Dict[str, torch.Tensor]:
    """
    Build counterfactual candidate samples around ground-truth waypoints.

    The first candidate is always the exact ground-truth waypoint.
    The remaining candidates are sampled by Gaussian translation noise.
    """

    if wpt_local.ndim != 2 or wpt_local.shape[1] != 3:
        raise ValueError(f"wpt_local must have shape (B, 3), got {wpt_local.shape}.")

    batch_size = wpt_local.shape[0]
    device = wpt_local.device
    dtype = wpt_local.dtype

    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2.")

    delta_xyz = torch.randn(
        batch_size,
        num_candidates,
        3,
        device=device,
        dtype=dtype,
    ) * float(trans_sigma)

    delta_xyz[:, 0, :] = 0.0

    candidates_xyz = wpt_local.unsqueeze(1) + delta_xyz
    trans_error = torch.linalg.vector_norm(delta_xyz, dim=-1)

    success_target = (trans_error <= float(success_radius)).float()

    nearest_dist = nearest_pc_distance(
        candidates_xyz=candidates_xyz,
        pc_list=pc_list,
        max_points=max_points,
    )

    collision_target = (nearest_dist <= float(collision_radius)).float()
    residual_target = wpt_local.unsqueeze(1) - candidates_xyz

    safe_sigma = max(float(trans_sigma), 1e-6)

    normalized_delta = delta_xyz / safe_sigma
    zero_rot_delta = torch.zeros_like(normalized_delta)
    pose_feat = torch.cat(
        [
            normalized_delta,
            zero_rot_delta,
            candidates_xyz,
        ],
        dim=-1,
    )

    return {
        "cand_pose_feat": pose_feat,
        "success_target": success_target,
        "collision_target": collision_target,
        "residual_target": residual_target,
        "candidates_xyz": candidates_xyz,
        "nearest_dist": nearest_dist,
    }


def build_ccf_inference_batch(
    center_xyz: torch.Tensor,
    num_candidates: int,
    trans_sigma: float,
) -> Dict[str, torch.Tensor]:
    """
    Build inference candidates around the current predicted waypoint.
    """

    if center_xyz.ndim != 2 or center_xyz.shape[1] != 3:
        raise ValueError(f"center_xyz must have shape (B, 3), got {center_xyz.shape}.")

    batch_size = center_xyz.shape[0]
    device = center_xyz.device
    dtype = center_xyz.dtype

    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2.")

    delta_xyz = torch.randn(
        batch_size,
        num_candidates,
        3,
        device=device,
        dtype=dtype,
    ) * float(trans_sigma)

    delta_xyz[:, 0, :] = 0.0

    candidates_xyz = center_xyz.unsqueeze(1) + delta_xyz

    safe_sigma = max(float(trans_sigma), 1e-6)
    normalized_delta = delta_xyz / safe_sigma
    zero_rot_delta = torch.zeros_like(normalized_delta)

    cand_pose_feat = torch.cat(
        [
            normalized_delta,
            zero_rot_delta,
            candidates_xyz,
        ],
        dim=-1,
    )

    return {
        "cand_pose_feat": cand_pose_feat,
        "candidates_xyz": candidates_xyz,
    }


def compute_ccf_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    success_weight: float,
    collision_weight: float,
    residual_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute CCF losses.
    """

    success_loss = F.binary_cross_entropy_with_logits(
        pred["success_logit"],
        target["success_target"],
    )

    collision_loss = F.binary_cross_entropy_with_logits(
        pred["collision_logit"],
        target["collision_target"],
    )

    residual_loss = F.smooth_l1_loss(
        pred["delta_xyz"],
        target["residual_target"],
    )

    total_loss = (
        float(success_weight) * success_loss
        + float(collision_weight) * collision_loss
        + float(residual_weight) * residual_loss
    )

    log = {
        "ccf_loss": float(total_loss.detach().cpu()),
        "ccf_success_loss": float(success_loss.detach().cpu()),
        "ccf_collision_loss": float(collision_loss.detach().cpu()),
        "ccf_residual_loss": float(residual_loss.detach().cpu()),
    }

    return total_loss, log


@torch.no_grad()
def refine_waypoint_with_ccf(
    ccf_head: CounterfactualContactField,
    scene_feat: torch.Tensor,
    center_xyz: torch.Tensor,
    num_candidates: int,
    trans_sigma: float,
    collision_penalty: float,
    max_residual: float,
) -> torch.Tensor:
    """
    Refine local waypoint prediction using the learned CCF head.
    """

    infer_batch = build_ccf_inference_batch(
        center_xyz=center_xyz,
        num_candidates=num_candidates,
        trans_sigma=trans_sigma,
    )

    pred = ccf_head(
        scene_feat=scene_feat,
        cand_pose_feat=infer_batch["cand_pose_feat"],
    )

    success_score = torch.sigmoid(pred["success_logit"])
    collision_score = torch.sigmoid(pred["collision_logit"])

    score = success_score - float(collision_penalty) * collision_score
    best_idx = score.argmax(dim=1)

    batch_idx = torch.arange(center_xyz.shape[0], device=center_xyz.device)

    best_xyz = infer_batch["candidates_xyz"][batch_idx, best_idx]
    best_delta = pred["delta_xyz"][batch_idx, best_idx]

    best_delta = torch.clamp(
        best_delta,
        min=-float(max_residual),
        max=float(max_residual),
    )

    refined_xyz = best_xyz + best_delta

    return refined_xyz