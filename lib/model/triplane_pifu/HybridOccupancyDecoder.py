import torch
import torch.nn as nn

from ..SurfaceClassifier import SurfaceClassifier


class HybridOccupancyDecoder(nn.Module):
    """
    将像素特征、三平面特征和深度特征融合，
    预测每个三维查询点的 Occupancy。

    对应核心函数：
        s(X | I) = f_theta(
            F_pixel(pi(X)),
            F_tri(X),
            z(X)
        )

    输入：
        pixel_feat:
            [B*V, pixel_dim, N]
            从 PIFu 图像特征图采样得到的像素对齐特征。

        tri_feat:
            [B*V, 3 * plane_dim, N]
            TriplaneSampler 输出的三平面特征。
            通道顺序为 xy、xz、yz。

        depth_feat:
            [B*V, 1, N]
            经过 PIFu DepthNormalizer 处理的深度特征。

    输出：
        occupancy:
            [B, 1, N]
            每个查询点位于物体内部的预测概率。

    注意：
        三个输入必须对应同一批查询点，且点的排列顺序一致。
        V 个视图必须连续存放；view_mask 只用于视图聚合，不是第四类特征。

        本模块只负责特征融合与 Occupancy 预测。
        坐标投影、平面采样、深度归一化和空间掩码，
        均由外部模块负责。
    """

    def __init__(
        self,
        pixel_dim=256,
        plane_dim=32,
        hidden_dims=(1024, 512, 256, 128),
        no_residual=False,
        num_views=1,
    ):
        """
        参数：
            pixel_dim:
                像素特征的通道数。

            plane_dim:
                单个特征平面的通道数。
                不是三个平面拼接后的总通道数。

            hidden_dims:
                点级 MLP 的隐藏层宽度。

            no_residual:
                False：使用 PIFu 原有的输入拼接跳连。
                True：使用普通逐层 MLP。
        """
        super().__init__()

        if type(pixel_dim) is not int or pixel_dim <= 0:
            raise ValueError("pixel_dim must be a positive integer.")

        if type(plane_dim) is not int or plane_dim <= 0:
            raise ValueError("plane_dim must be a positive integer.")

        if not isinstance(hidden_dims, (tuple, list)) or not hidden_dims:
            raise ValueError(
                "hidden_dims must be a nonempty tuple or list."
            )

        if any(type(dim) is not int or dim <= 0 for dim in hidden_dims):
            raise ValueError(
                "Every hidden dimension must be a positive integer."
            )

        if not isinstance(no_residual, bool):
            raise TypeError("no_residual must be a boolean.")

        self.pixel_dim = pixel_dim
        self.plane_dim = plane_dim

        # Sampler 将 xy、xz、yz 三个平面的采样结果按通道拼接。
        self.tri_dim = 3 * plane_dim

        # PIFu 深度特征是每个查询点一个标量。
        self.depth_dim = 1

        # 默认：256 + 3 * 32 + 1 = 353。
        self.input_dim = (
            self.pixel_dim
            + self.tri_dim
            + self.depth_dim
        )

        # 复用已有 PIFu 的点级 MLP。
        # 修改输入维度，使其能够接收三类融合特征。
        #
        # 按原 PIFu 的方式在 MLP 中间聚合视图，输出物体批次 B。
        # 最后的 Sigmoid 将输出转换为 Occupancy 概率。
        self.classifier = SurfaceClassifier(
            filter_channels=[
                self.input_dim,
                *hidden_dims,
                1,
            ],
            num_views=num_views,
            no_residual=no_residual,
            last_op=nn.Sigmoid(),
        )

    def _validate_features(
        self,
        pixel_feat,
        tri_feat,
        depth_feat,
    ):
        """检查输入类型、形状、通道数和设备。"""
        inputs = (
            ("pixel_feat", pixel_feat, self.pixel_dim),
            ("tri_feat", tri_feat, self.tri_dim),
            ("depth_feat", depth_feat, self.depth_dim),
        )

        for name, feature, expected_channels in inputs:
            if not isinstance(feature, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor."
                )

            if feature.ndim != 3:
                raise ValueError(
                    f"{name} must have shape [B, C, N]."
                )

            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"{name}: expected {expected_channels} channels, "
                    f"got {feature.shape[1]}."
                )

            if feature.shape[0] == 0 or feature.shape[2] == 0:
                raise ValueError(
                    f"{name} must contain a nonempty batch and points."
                )

            if not feature.is_floating_point():
                raise TypeError(
                    f"{name} must be floating point."
                )

        expected_shape = (
            pixel_feat.shape[0],
            pixel_feat.shape[2],
        )

        for name, feature in (
            ("tri_feat", tri_feat),
            ("depth_feat", depth_feat),
        ):
            actual_shape = (
                feature.shape[0],
                feature.shape[2],
            )

            if actual_shape != expected_shape:
                raise ValueError(
                    f"{name} must have the same batch size "
                    "and point count as pixel_feat."
                )

            if feature.device != pixel_feat.device:
                raise ValueError(
                    "All feature tensors must be on the same device."
                )

    def forward(
        self,
        pixel_feat,
        tri_feat,
        depth_feat,
        view_mask=None,
    ):
        """
        对同一组查询点进行特征融合和 Occupancy 预测。

        返回：
            occupancy: [B, 1, N]
        """
        self._validate_features(
            pixel_feat,
            tri_feat,
            depth_feat,
        )

        # 沿通道维拼接，查询点维度 N 保持不变。
        #
        # 默认：
        # [B, 256, N]
        # [B,  96, N]
        # [B,   1, N]
        #       ↓
        # [B, 353, N]
        fused_feat = torch.cat(
            [
                pixel_feat,
                tri_feat,
                depth_feat,
            ],
            dim=1,
        )

        # SurfaceClassifier 内部已经包含最后的 Sigmoid。
        occupancy = self.classifier(fused_feat, view_mask=view_mask)

        return occupancy
