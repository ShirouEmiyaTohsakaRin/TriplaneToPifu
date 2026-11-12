import torch


def transform_uv(uv, transforms):
    """Apply a shared [2,3] or per-view [B*V,2,3] normalized-image affine."""
    if transforms is None:
        return uv
    transforms = transforms.to(device=uv.device, dtype=uv.dtype)
    if transforms.shape == (2, 3):
        transforms = transforms.unsqueeze(0).expand(uv.shape[0], -1, -1)
    if transforms.shape != (uv.shape[0], 2, 3):
        raise ValueError('transforms must be [2,3] or [B*V,2,3].')
    return torch.baddbmm(transforms[:, :, 2:3], transforms[:, :, :2], uv)


def index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    return torch.cat([transform_uv(pts[:, :2], transforms), pts[:, 2:3]], dim=1)


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    depth = homo[:, 2:3, :]
    # Keep points on the camera plane finite; callers mask nonpositive depth.
    denominator = torch.where(depth < 0, -depth.abs().clamp_min(1e-6), depth.abs().clamp_min(1e-6))
    xy = transform_uv(homo[:, :2, :] / denominator, transforms)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz
