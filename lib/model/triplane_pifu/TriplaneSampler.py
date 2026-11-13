import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneSampler(nn.Module):
    """
    目的：按点的位置，从三个平面各取一份特征，交给后面的网络判断内外。
    每个平面取 32 个数时，合起来就是这个点的 96 个数。

    先调用 normalize_coordinates() 换算深度，再调用 forward() 取值。
    三个平面分别用 (u,v)、(u,换算后的深度)、(v,换算后的深度) 查找。
    planes：[B, 3, C, H, W]；q：[B, 3, N]；输出：[B, 3*C, N]。
    B 是图像总数，C 是每个位置的特征长度，N 是点数。

    任一坐标超出 [-1,1]，该点从三个平面取到的特征全部设为 0。
    这只表示超出了平面范围，不能据此判断点一定在物体外部。
    """

    def __init__(self, depth_center, depth_radius):
        super().__init__()

        depth_center = float(depth_center)
        depth_radius = float(depth_radius)

        if not math.isfinite(depth_center):
            raise ValueError(
                "depth_center must be finite."
            )

        if (
            not math.isfinite(depth_radius)
            or depth_radius <= 0
        ):
            raise ValueError(
                "depth_radius must be finite and positive."
            )

        # center 和 radius 决定深度怎样换算，训练时不修改它们。
        # 随模型一起保存，重建时才能使用相同的换算方式。
        self.register_buffer(
            "depth_center",
            torch.tensor(
                depth_center,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "depth_radius",
            torch.tensor(
                depth_radius,
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _validate_coordinates(coordinates, name):
        if not isinstance(coordinates, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if (
            coordinates.ndim != 3
            or coordinates.shape[1] != 3
        ):
            raise ValueError(
                f"{name} must have shape [B, 3, N]."
            )

        if (
            coordinates.shape[0] == 0
            or coordinates.shape[2] == 0
        ):
            raise ValueError(
                f"{name} must contain a nonempty batch and points."
            )

        if not coordinates.is_floating_point():
            raise TypeError(
                f"{name} must be floating point."
            )

        if not torch.isfinite(coordinates).all():
            raise ValueError(
                f"{name} contains NaN or infinity."
            )

    def normalize_coordinates(self, projected_xyz):
        """
        目的：把深度换算到平面使用的坐标尺度，才能找到正确的取值位置。
        输入 projected_xyz：[B, 3, N]，依次是图像位置 u、v 和相机投影后的深度 d。

        u、v 不变；深度按 (d - depth_center) / depth_radius 换算。
        例如中心为 10、半宽为 2，深度 8、10、12 就对应 -1、0、1。
        超出范围的数值不截断，后面才能识别越界点。

        返回 q=(u,v,换算后的深度)，形状不变。
        请传入投影后的坐标，d 不要提前经过 DepthNormalizer，也不要重复换算。
        """
        self._validate_coordinates(
            projected_xyz,
            "projected_xyz",
        )

        center = self.depth_center.to(projected_xyz)
        radius = self.depth_radius.to(projected_xyz)

        uv = projected_xyz[:, :2, :]

        depth = (
            projected_xyz[:, 2:3, :] - center
        ) / radius

        # 保留超出 [-1, 1] 的坐标，后面才能识别哪些点越界。
        return torch.cat(
            [uv, depth],
            dim=1,
        )

    def forward(self, planes, q, return_mask=False):
        """
        目的：取出一个点在三个平面上的特征，并按 xy、xz、yz 的顺序拼起来。
        点落在格点之间时，用周围四个格点的数值算出结果。

        planes：[B, 3, C, H, W]；q：[B, 3, N]，是换算好的点坐标。
        返回 features：[B, 3*C, N]，每个点得到 3*C 个数。
        return_mask=True 时，还返回 [B, 1, N] 的 valid_mask。
        True 表示三个坐标都在 [-1,1] 内，False 表示越界。
        """
        self._validate_coordinates(q, "q")

        if not isinstance(planes, torch.Tensor):
            raise TypeError(
                "planes must be a torch.Tensor."
            )

        if planes.ndim != 5 or planes.shape[1] != 3:
            raise ValueError(
                "planes must have shape [B, 3, C, H, W]."
            )

        if not planes.is_floating_point():
            raise TypeError(
                "planes must be floating point."
            )

        if planes.shape[0] != q.shape[0]:
            raise ValueError(
                "planes and q must have the same batch size."
            )

        if planes.device != q.device:
            raise ValueError(
                "planes and q must be on the same device."
            )

        batch_size, _, channels, height, width = planes.shape

        if channels < 1 or height < 2 or width < 2:
            raise ValueError(
                "planes require C >= 1 and H, W >= 2."
            )

        num_points = q.shape[2]

        # 1. 标记哪些点超出平面的表示范围，稍后将这些点的全部三平面特征清零。
        # [B, 1, N]
        valid_mask = (
            (q >= -1.0) & (q <= 1.0)
        ).all(dim=1, keepdim=True)

        # safe_q 临时把越界点移到 0，避免把越界坐标直接交给 grid_sample。
        # 最后仍用原来的 valid_mask 清零，不能把临时取到的特征当成有效结果。
        safe_q = q.masked_fill(
            ~valid_mask,
            0.0,
        )

        # 2. 每个平面只能接收两个坐标，所以把点拆成 (u,v)、(u,深度)、(v,深度)。
        # 每对坐标按“横、纵”排列，与平面生成时的方向保持一致。
        # [B, 3, 2, N]
        grids = torch.stack(
            [
                safe_q[:, [0, 1], :],  # xy
                safe_q[:, [0, 2], :],  # xz
                safe_q[:, [1, 2], :],  # yz
            ],
            dim=1,
        )

        # 3. 把平面排到一起，一次完成取值；对应坐标也要按相同顺序排列。

        # [B, 3, 2, N] -> [B, 3, N, 2]
        #              -> [B*3, N, 1, 2]
        grids = grids.permute(
            0, 1, 3, 2
        ).reshape(
            batch_size * 3,
            num_points,
            1,
            2,
        )

        grids = grids.to(dtype=planes.dtype)

        # [B, 3, C, H, W] -> [B*3, C, H, W]
        plane_batch = planes.reshape(
            batch_size * 3,
            channels,
            height,
            width,
        )

        # 4. 查询位置落在格点之间时，用周围四点的特征按位置比例算出结果。
        # 这种计算也允许预测误差传回生成器，让它学习生成更合适的平面。
        # 输出：[B*3, C, N, 1]
        # 用 -1 和 1 表示两端格点的中心，让这里的取值位置与生成平面时一致。
        sampled = F.grid_sample(
            plane_batch,
            grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # 5. 把同一点的三份平面特征接起来，交给预测网络；默认共 96 个数。

        # [B*3, C, N, 1] -> [B, 3, C, N]
        sampled = sampled.squeeze(-1).reshape(
            batch_size,
            3,
            channels,
            num_points,
        )

        # [B, 3, C, N] -> [B, 3*C, N]
        features = sampled.reshape(
            batch_size,
            3 * channels,
            num_points,
        )

        # 越界点取到的特征全部清零，不使用这些结果，也不让它们影响生成器的学习。
        features = features.masked_fill(
            ~valid_mask,
            0.0,
        )

        if return_mask:
            return features, valid_mask

        return features
