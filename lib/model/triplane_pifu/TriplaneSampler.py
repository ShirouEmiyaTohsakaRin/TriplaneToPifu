import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneSampler(nn.Module):
    """
    按查询点的位置，从三个特征平面中取值，再拼接成一组特征。

    先调用 normalize_coordinates()，把投影坐标 (u, v, d) 转成
    q = (u, v, 归一化深度)，再调用 forward(planes, q) 取特征。
    三个平面依次为 xy、xz、yz，对应 (u, v)、(u, 深度)、(v, 深度)。
    每对坐标的第一个值是横坐标，第二个值是纵坐标。

    planes: [B, 3, C, H, W]；q: [B, 3, N]；输出: [B, 3*C, N]。
    这里 B 是输入视图总数，C 是每个平面的特征维度，N 是查询点数。
    q 的任一坐标超出 [-1, 1] 时，该点的三平面特征全部设为 0。
    越界时清零只表示这里取不到平面特征，不能据此判断该点在物体外部。
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

        # 深度范围在训练中保持不变，不作为模型参数学习。
        # 把它们随模型一起放到 CPU 或 GPU，并保存到权重文件中。
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
        保留图像位置 u、v，把深度换算成平面使用的坐标。

        输入 projected_xyz: [B, 3, N]，三个通道依次是 u、v、投影深度 d。
        d 必须是经过相机变换、但尚未经过 PIFu DepthNormalizer 的深度。
        透视投影下，u、v 已经做过除以深度的运算。

        返回 q = (u, v, (d - depth_center) / depth_radius)。
        center 是深度范围的中心，radius 是范围宽度的一半。
        例如 d = center 对应 0，d = center + radius 对应 1。
        输入必须是投影后的坐标，不能传原始世界坐标或重复归一化后的 q。
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
        从三个平面分别取特征，并按 xy、xz、yz 的顺序拼接。

        planes: [B, 3, C, H, W]，三个特征平面。
        q: [B, 3, N]，由 normalize_coordinates() 得到的查询坐标。
        返回 features: [B, 3*C, N]，每个点共得到 3*C 个特征数值。

        return_mask=True 时，还返回 valid_mask: [B, 1, N]。
        True 表示该点的三个坐标都在 [-1, 1] 内，False 表示越界。
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
        # 步骤 1：检查每个点的三个坐标是否都在 [-1, 1] 内
        # [B, 1, N]
        # --------------------------------------------------
        valid_mask = (
            (q >= -1.0) & (q <= 1.0)
        ).all(dim=1, keepdim=True)

        # 越界点先用坐标 0 代替，让后面的取值操作正常执行。
        # 这些点取到的特征会在最后清零，不参与最终特征输出。
        safe_q = q.masked_fill(
            ~valid_mask,
            0.0,
        )

        # --------------------------------------------------
        # 步骤 2：分别用 (u, v)、(u, 深度)、(v, 深度) 到三个平面上找位置
        # 每一对坐标依次是横坐标、纵坐标。
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
        # 步骤 3：把三个平面当作独立样本，连同查询坐标一起排好，方便一次取值
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
        # 步骤 4：按位置对周围四个网格点加权，得到该位置的特征（双线性插值）
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
        # 步骤 5：把结果按样本分组，再把三个平面的特征拼在一起
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

        # 只要点的任一坐标越界，就把它从三个平面取到的特征全部设为 0。
        features = features.masked_fill(
            ~valid_mask,
            0.0,
        )

        if return_mask:
            return features, valid_mask

        return features
