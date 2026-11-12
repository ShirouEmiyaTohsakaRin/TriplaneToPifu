import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneSampler(nn.Module):
    """
    从 TriplaneGenerator 输出的三个平面中采样特征。

    平面顺序：
        0: xy，横轴为 x，纵轴为 y
        1: xz，横轴为 x，纵轴为 z
        2: yz，横轴为 y，纵轴为 z

    坐标转换：
        normalize_coordinates(projected_xyz)

        输入为 PIFu 投影后的 (u,v,d) 坐标 [B*V, 3, N]，
        输出为三平面采样坐标 q。

    特征采样：
        forward(planes, q)

        planes: [B, 3, C, H, W]
        q:      [B, 3, N]，已经归一化
        output: [B, 3*C, N]

    三平面有效空间为 [-1, 1]^3。
    越界点的三平面特征设为零，但不强制其 occupancy 为零。
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

        # 固定归一化参数，不参与梯度更新。
        # 随模型移动设备，并保存在 state_dict 中。
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
        将 PIFu 投影后的坐标转换为三平面采样坐标。
        透视模式的 u/v 已经除以深度；这是视锥参数化，不是世界 XYZ。

        输入：
            projected_xyz: [B, 3, N]

            前两个通道：
                图像采样坐标 u、v。

            第三个通道：
                标定变换后的深度 d，
                尚未经过 PIFu DepthNormalizer。

        转换：
            q = (u, v, (d - depth_center) / depth_radius)

        不应传入：
            1. 原始世界坐标；
            2. 已经归一化过的 q。
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

        # 不截断坐标，保留越界信息供 forward() 判断。
        return torch.cat(
            [uv, depth],
            dim=1,
        )

    def forward(self, planes, q, return_mask=False):
        """
        参数：
            planes: [B, 3, C, H, W]
            q: [B, 3, N]

            return_mask:
                是否同时返回三平面有效空间掩码。

        返回：
            features: [B, 3*C, N]
                特征按 xy、xz、yz 顺序拼接。

            valid_mask: [B, 1, N]，可选
                True 表示查询点位于 [-1, 1]^3 内。
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

        # --------------------------------------------------
        # Step 1：判断点是否处于三平面的有效三维范围内
        # [B, 1, N]
        # --------------------------------------------------
        valid_mask = (
            (q >= -1.0) & (q <= 1.0)
        ).all(dim=1, keepdim=True)

        # 越界点临时使用安全坐标进行计算。
        # 这些点的输出会在最后统一清零。
        safe_q = q.masked_fill(
            ~valid_mask,
            0.0,
        )

        # --------------------------------------------------
        # Step 2：为三个平面分别构造二维查询坐标
        # 每一对坐标都是“横坐标、纵坐标”
        # [B, 3, 2, N]
        # --------------------------------------------------
        grids = torch.stack(
            [
                safe_q[:, [0, 1], :],  # xy
                safe_q[:, [0, 2], :],  # xz
                safe_q[:, [1, 2], :],  # yz
            ],
            dim=1,
        )

        # --------------------------------------------------
        # Step 3：按照相同顺序合并批次和平面维度
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Step 4：双线性插值
        # 输出：[B*3, C, N, 1]
        # --------------------------------------------------
        sampled = F.grid_sample(
            plane_batch,
            grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # --------------------------------------------------
        # Step 5：恢复平面维度，再拼接三个平面的通道
        # --------------------------------------------------

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

        # 三个平面共同遵守同一个三维有效范围。
        features = features.masked_fill(
            ~valid_mask,
            0.0,
        )

        if return_mask:
            return features, valid_mask

        return features
