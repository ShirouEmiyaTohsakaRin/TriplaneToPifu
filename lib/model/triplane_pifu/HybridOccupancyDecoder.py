import torch
import torch.nn as nn

from ..SurfaceClassifier import SurfaceClassifier


class HybridOccupancyDecoder(nn.Module):
    """
    目的：把图像特征、三平面特征和深度放在一起，判断点是否在物体内部。
    网络通过训练学习如何使用这三份依据，最后输出 0～1 之间的概率。
    越接近 1，越倾向于判断点在内部；真实标签中，内部为 1，外部为 0。

    B 是物体数，V 是每个物体的图像数，N 是点数。
    pixel_feat：[B*V, pixel_dim, N]，图像特征。
    tri_feat：[B*V, 3*plane_dim, N]，三个平面取到的特征。
    depth_feat：[B*V, 1, N]，按 PIFu 方式缩放的深度。
    输出 occupancy：[B, 1, N]，每个物体的每个点只有一份预测。

    三份输入的点顺序必须相同，同一物体的各张图像要放在一起。
    多视图时，按 view_mask 排除投影无效的视图，再在网络中间把其余视图的特征平均。
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
        目的：设置预测网络的大小，让它能接收三份输入。
        pixel_dim 是图像特征长度，plane_dim 是一个平面的特征长度。
        总输入长度为 pixel_dim + 3*plane_dim + 1，默认是 353。
        hidden_dims 设置中间各层的大小，num_views 设置每个物体有几张图。

        no_residual=False 时，后面的层还会再次收到原始输入，方便直接参考最初的依据。
        True 时，后面的层只接收上一层的结果。
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

        # 三个平面各提供 plane_dim 个数，合起来就是 3 倍。
        self.tri_dim = 3 * plane_dim

        # 深度保留为一个额外输入，帮助网络区分同一像素前后的点。
        self.depth_dim = 1

        # 把三份输入的长度加起来，默认是 256+96+1=353。
        self.input_dim = (
            self.pixel_dim
            + self.tri_dim
            + self.depth_dim
        )

        # 所有点使用同一套预测网络，各点的特征分别处理。
        # 多张图像对应同一点时，网络中间会合并它们，最后输出一个 0～1 的概率。
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
        """
        目的：提前发现输入尺寸或设备不匹配的问题，给出更明确的报错。
        这里只检查类型、尺寸和设备；点的顺序是否对应，仍要由调用者保证。
        """
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
        目的：把同一个点的三份输入拼起来，得到它位于物体内部的概率。
        各个点使用同一套预测网络，不把不同点的特征混在一起。
        输入多张图像时，合并的是同一物体、同一点在各视图下的特征。
        返回 [B, 1, N]，每个点对应一份概率。
        """
        self._validate_features(
            pixel_feat,
            tri_feat,
            depth_feat,
        )

        # 把同一点的三份输入接起来，让网络学习如何使用它们。
        # dim=1 增加的是每个点的输入长度，点的顺序不变；默认 [B*V, 353, N]。
        fused_feat = torch.cat(
            [
                pixel_feat,
                tri_feat,
                depth_feat,
            ],
            dim=1,
        )

        # 这个网络已经输出 0～1 的概率，直接返回，不要再做一次 Sigmoid 改变结果。
        occupancy = self.classifier(fused_feat, view_mask=view_mask)

        return occupancy
