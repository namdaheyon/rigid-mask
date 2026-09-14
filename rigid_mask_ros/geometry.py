"""Small pure-PyTorch geometry subset replacing the Kornia dependency."""

import torch


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3) Rodrigues vectors to (..., 3, 3) matrices.

    The implementation is differentiable, device/dtype preserving, and stable
    for rotations close to zero. RigidMask only needs the inference path.
    """
    if angle_axis.shape[-1] != 3:
        raise ValueError("angle_axis must have shape (..., 3)")

    x, y, z = angle_axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(angle_axis.shape[:-1] + (3, 3))

    theta2 = (angle_axis * angle_axis).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(torch.finfo(angle_axis.dtype).eps))
    theta2_matrix = theta2.unsqueeze(-1)
    theta_matrix = theta.unsqueeze(-1)

    # Taylor expansions avoid 0/0 while keeping useful gradients near zero.
    sin_over_theta = torch.where(
        theta2_matrix > 1e-8,
        torch.sin(theta_matrix) / theta_matrix,
        1.0 - theta2_matrix / 6.0 + theta2_matrix * theta2_matrix / 120.0,
    )
    one_minus_cos_over_theta2 = torch.where(
        theta2_matrix > 1e-8,
        (1.0 - torch.cos(theta_matrix)) / theta2_matrix,
        0.5 - theta2_matrix / 24.0 + theta2_matrix * theta2_matrix / 720.0,
    )
    identity = torch.eye(3, dtype=angle_axis.dtype, device=angle_axis.device)
    identity = identity.expand(angle_axis.shape[:-1] + (3, 3))
    return identity + sin_over_theta * skew + one_minus_cos_over_theta2 * (skew @ skew)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first (..., 4) quaternions to rotation matrices."""
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have shape (..., 4)")
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    two = q.new_tensor(2.0)
    values = (
        1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
        two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
        two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
    )
    return torch.stack(values, dim=-1).reshape(quaternion.shape[:-1] + (3, 3))
