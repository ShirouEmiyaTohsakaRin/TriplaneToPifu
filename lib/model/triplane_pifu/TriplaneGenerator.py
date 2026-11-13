import torch
import torch.nn as nn


def make_grid_coordinates(resolution):
    """
    目的：给每个网格位置配上坐标，后面才能告诉网络“这是哪个位置”。
    resolution 表示每行、每列各有几个点，坐标从 -1 到 1 均匀排列。
    例如 resolution=2，得到四个点：(-1,-1)、(1,-1)、(-1,1)、(1,1)。
    返回 [resolution*resolution, 2]，每行依次是横坐标、纵坐标。
    """
    axis = torch.linspace(
        -1.0,
        1.0,
        resolution,
    )

    vertical, horizontal = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    return torch.stack(
        [horizontal, vertical],
        dim=-1,
    ).reshape(-1, 2)


class TriplaneGenerator(nn.Module):
    """
    目的：根据图像生成三个特征平面，供后面的网络按三维点的位置查找。
    特征是网络提取的一组数，用来表示图像或空间中的内容。
    加入深度方向的平面后，同一像素前后的点就可以取到不同的特征。

    输入 image_features：[B, image_feat_dim, H, W]，从图像提取的特征。
    输出 planes：[B, 3, triplane_dim, R, R]，每张图对应三个平面。
    B 是图像数，R 是平面边长，triplane_dim 是每个位置的特征长度。
    三个平面的横、纵轴依次是 (u,v)、(u,换算后的深度)、(v,换算后的深度)。
    输入多张图像时，用同一个生成器分别生成各自的平面。
    """

    def __init__(
        self,
        image_feat_dim=256,
        inner_dim=256,
        image_token_res=16,
        triplane_low_res=16,
        triplane_high_res=64,
        triplane_dim=32,
        num_layers=4,
        num_heads=8,
        dropout=0.0,
        eps=1e-6,
    ):
        super().__init__()

        sizes = (
            image_feat_dim,
            inner_dim,
            image_token_res,
            triplane_low_res,
            triplane_high_res,
            triplane_dim,
            num_layers,
            num_heads,
        )

        if any(
            type(value) is not int or value <= 0
            for value in sizes
        ):
            raise ValueError(
                "Dimensions and layer counts must be positive integers."
            )

        if image_token_res < 2 or triplane_low_res < 2:
            raise ValueError(
                "Image and plane grid resolutions must be >= 2."
            )

        if triplane_high_res < triplane_low_res:
            raise ValueError(
                "triplane_high_res must be >= triplane_low_res."
            )

        if inner_dim % num_heads != 0:
            raise ValueError(
                "inner_dim must be divisible by num_heads."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if not eps > 0:
            raise ValueError(
                "eps must be positive."
            )

        self.image_feat_dim = image_feat_dim
        self.inner_dim = inner_dim
        self.image_token_res = image_token_res
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # 坐标固定不变，不需要训练；随模型移到 CPU 或 GPU，加载时可以重新生成。
        self.register_buffer(
            "image_coordinates",
            make_grid_coordinates(image_token_res),
            persistent=False,
        )

        self.register_buffer(
            "plane_coordinates",
            make_grid_coordinates(triplane_low_res),
            persistent=False,
        )

        # 1. 先缩小特征图，减少后面要处理的位置，节省计算。
        self.image_pool = nn.AdaptiveAvgPool2d(
            image_token_res
        )

        # 把每个位置的特征长度调整为 inner_dim，让后面的网络能够接收。
        self.image_projection = nn.Conv2d(
            image_feat_dim,
            inner_dim,
            kernel_size=1,
        )

        # 把坐标也加入图像特征，让网络知道每份特征来自哪里。
        self.image_position = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        # 调整特征数值的大小，避免数值尺度差异过大，便于后续计算。
        self.memory_norm = nn.LayerNorm(
            inner_dim,
            eps=eps,
        )

        # 2. 为每个平面位置准备起始特征，之后根据图像更新它。

        # 给不同格点加上各自的位置，避免网络分不清它们。
        self.query_position = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        # 给三个平面各加一个可学习的标记，区分 xy、xz、yz。
        self.plane_embedding = nn.Embedding(
            3,
            inner_dim,
        )

        nn.init.normal_(
            self.plane_embedding.weight,
            std=0.02,
        )

        # 加入当前图像的整体特征，让平面的起始特征也随图像变化。
        self.image_condition = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.query_norm = nn.LayerNorm(
            inner_dim,
            eps=eps,
        )

        # 3. 让网络学习：每个平面位置应该参考图像的哪些区域。

        # 每层先调整数值大小，再读取图像和其他位置的特征，逐步更新平面。
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=inner_dim,
                nhead=num_heads,
                dim_feedforward=4 * inner_dim,
                dropout=dropout,
                activation="gelu",
                layer_norm_eps=eps,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(
            inner_dim,
            eps=eps,
        )

        # 4. 先在小平面上计算，再放大平面，避免一开始就处理大量位置。

        # 每个位置参考周围格点来调整特征；三个平面使用同一套处理网络。
        head_dim = max(
            triplane_dim,
            inner_dim // 4,
        )

        readout = [
            nn.Conv2d(
                inner_dim,
                head_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        ]

        resolution = triplane_low_res

        while resolution < triplane_high_res:
            resolution = min(
                2 * resolution,
                triplane_high_res,
            )

            # 先根据已有格点补出更多位置，再让网络学习怎样调整这些位置的特征。
            readout.extend([
                nn.Upsample(
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
            ])

        # 这里输出的特征还要继续用于预测，因此允许有正有负，不限制在 0～1。
        readout.append(
            nn.Conv2d(
                head_dim,
                triplane_dim,
                kernel_size=1,
            )
        )

        self.plane_readout = nn.Sequential(*readout)

    def forward(self, image_features):
        if image_features.ndim != 4:
            raise ValueError(
                "image_features must have shape [B, C, H, W]."
            )

        if image_features.shape[1] != self.image_feat_dim:
            raise ValueError(
                f"Expected {self.image_feat_dim} input channels, "
                f"got {image_features.shape[1]}."
            )

        if min(
            image_features.shape[0],
            *image_features.shape[-2:],
        ) < 1:
            raise ValueError(
                "Batch size and spatial dimensions must be positive."
            )

        if not image_features.is_floating_point():
            raise TypeError(
                "image_features must be floating point."
            )

        batch_size = image_features.shape[0]
        low_res = self.triplane_low_res
        tokens_per_plane = low_res ** 2

        # 1. 缩小特征图，减少计算；默认保留 16×16 个位置，每处存 256 个数。
        # [B, C, H, W] -> [B, D, S, S]
        projected = self.image_projection(
            self.image_pool(image_features)
        )

        # 2. 把图像特征排成一列，并加上位置，供后面的平面生成使用。
        # [B, D, S, S] -> [B, S*S, D]
        image_tokens = projected.flatten(2).transpose(1, 2)

        image_pos = self.image_position(
            self.image_coordinates.to(image_tokens)
        ).unsqueeze(0)

        memory = self.memory_norm(
            image_tokens + image_pos
        )

        # 3. 给每个平面格点准备起始特征：位置、平面编号和当前图像。

        # position 说明格点在哪里，例如左上角或右下角。[1, 1, L*L, D]
        position = self.query_position(
            self.plane_coordinates.to(image_tokens)
        ).reshape(
            1,
            1,
            tokens_per_plane,
            self.inner_dim,
        )

        # identity 说明格点属于哪个平面，避免三个平面的同一位置混淆。[1, 3, 1, D]
        identity = self.plane_embedding.weight.reshape(
            1,
            3,
            1,
            self.inner_dim,
        )

        # condition 是整张图像的总体特征，让网络知道这次要为哪张图生成平面。
        condition = self.image_condition(
            projected.mean(dim=(2, 3))
        ).reshape(
            batch_size,
            1,
            1,
            self.inner_dim,
        )

        # 把这三部分相加，得到每个格点的起始特征 queries，接下来再更新。
        # 结果形状：[B, 3, L*L, D]。
        queries = position + identity + condition

        queries = queries.reshape(
            batch_size,
            3 * tokens_per_plane,
            self.inner_dim,
        )

        queries = self.query_norm(queries)

        # 4. 每个格点先参考其他格点，再从图像中选取相关特征，更新自己的内容。
        for layer in self.decoder_layers:
            queries = layer(
                tgt=queries,
                memory=memory,
            )

        queries = self.output_norm(queries)

        # 5. 把算好的格点特征排回三个平面，后面就能按坐标查找。

        # [B, 3*L*L, D] -> [B, 3, L, L, D]
        planes = queries.reshape(
            batch_size,
            3,
            low_res,
            low_res,
            self.inner_dim,
        )

        # [B, 3, L, L, D] -> [B, 3, D, L, L]
        planes = planes.permute(
            0, 1, 4, 2, 3
        ).contiguous()

        # 把三个平面当作三张特征图交给同一个网络处理，完成后再放回原来的分组。
        # [B, 3, D, L, L] -> [B*3, D, L, L]
        planes = planes.reshape(
            batch_size * 3,
            self.inner_dim,
            low_res,
            low_res,
        )

        # 6. 默认把平面从 16×16 放大到 64×64，每处存 32 个数，供查询点取值。
        planes = self.plane_readout(planes)

        return planes.reshape(
            batch_size,
            3,
            self.triplane_dim,
            self.triplane_high_res,
            self.triplane_high_res,
        ).contiguous()
