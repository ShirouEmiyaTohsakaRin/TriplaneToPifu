import torch
import torch.nn as nn

from ..SurfaceClassifier import SurfaceClassifier


class HybridOccupancyDecoder(nn.Module):
    """
    把三类特征拼在一起，预测每个点位于物体内部的概率（Occupancy）。

    B 是物体数，V 是每个物体的视图数，N 是查询点数。
    输入特征由外部准备好，本模块只负责拼接和预测：
        pixel_feat: [B*V, pixel_dim, N]，在点的图像投影位置取到的特征。
        tri_feat: [B*V, 3*plane_dim, N]，从 xy、xz、yz 三个平面取到的特征。
        depth_feat: [B*V, 1, N]，经过 PIFu DepthNormalizer 处理的深度。
    输出 occupancy: [B, 1, N]，每个点对应一个 0 到 1 之间的预测值。

    三类特征的点顺序必须一致，同一物体的 V 个视图必须连续存放。
    view_mask 标记点在哪些视图的图像范围内，用于合并这些视图的信息。
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
        pixel_dim: 每个点从图像中取到多少维特征。
        plane_dim: 每个平面的特征维度，三个平面拼接后是它的 3 倍。
        hidden_dims: MLP 各隐藏层的特征维度。
        no_residual: False 时，后续层会再次拼入最初的输入特征；
            True 时，只使用上一层的输出。
        num_views: 每个物体输入多少个视图。
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

        # 每个平面提供 plane_dim 维特征，三个拼起来就是 3 倍。
        self.tri_dim = 3 * plane_dim

        # 每个查询点只需要一个深度数值。
        self.depth_dim = 1

        # 默认：256 + 3 * 32 + 1 = 353。
        self.input_dim = (
            self.pixel_dim
            + self.tri_dim
            + self.depth_dim
        )

        # 沿用 PIFu 的 MLP，把输入大小调整为三类特征的总维度。
        # 多视图时，在 MLP 中间合并同一物体各视图的信息。
        # 最后用 Sigmoid 得到 0～1 之间的内部概率。
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
        """检查三类特征的类型和尺寸是否符合要求，以及是否在同一设备上。"""
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
        拼接同一组点的三类特征，交给 MLP 预测。
        返回 occupancy: [B, 1, N]，表示各点位于物体内部的概率。
        """
        self._validate_features(
            pixel_feat,
            tri_feat,
            depth_feat,
        )

        # 将同一个点的三类特征拼在一起，点的数量和顺序保持不变。
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

        # SurfaceClassifier 最后已经做了 Sigmoid，这里直接使用它输出的概率。
        occupancy = self.classifier(fused_feat, view_mask=view_mask)

        return occupancy
