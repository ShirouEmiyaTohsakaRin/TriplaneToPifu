import torch
import torch.nn as nn


def make_grid_coordinates(resolution):
    """
    构造二维规则网格坐标。

    返回：
        [resolution * resolution, 2]

    每个坐标对的顺序：
        (水平坐标, 垂直坐标)

    坐标范围：
        [-1, 1]
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
    从 PIFu 图像特征生成视图对齐的三个特征平面。

    输入：
        image_features: [B, image_feat_dim, H, W]

    输出：
        planes: [B, 3, triplane_dim, R, R]

    平面顺序及存储方向：
        0: xy，横轴为 x，纵轴为 y
        1: xz，横轴为 x，纵轴为 z
        2: yz，横轴为 y，纵轴为 z

    查询构造：
        坐标编码 + 平面身份编码 + 图片全局条件

    注意力实现：
        PyTorch 标准 TransformerDecoderLayer。
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

        # 固定网格可以由配置重新生成，不作为学习参数保存。
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

        # --------------------------------------------------
        # 1. 将 PIFu 图像特征转换成图像 tokens
        # --------------------------------------------------
        self.image_pool = nn.AdaptiveAvgPool2d(
            image_token_res
        )

        self.image_projection = nn.Conv2d(
            image_feat_dim,
            inner_dim,
            kernel_size=1,
        )

        self.image_position = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.memory_norm = nn.LayerNorm(
            inner_dim,
            eps=eps,
        )

        # --------------------------------------------------
        # 2. 构造三平面查询
        # --------------------------------------------------

        # 平面内部二维坐标的编码。
        self.query_position = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        # 三个可学习向量，分别标识 xy、xz、yz。
        self.plane_embedding = nn.Embedding(
            3,
            inner_dim,
        )

        nn.init.normal_(
            self.plane_embedding.weight,
            std=0.02,
        )

        # 将当前图片的整体特征转换成查询条件。
        self.image_condition = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.query_norm = nn.LayerNorm(
            inner_dim,
            eps=eps,
        )

        # --------------------------------------------------
        # 3. 标准 Transformer 解码层
        # --------------------------------------------------

        # 分别创建各层，使它们独立初始化。
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

        # --------------------------------------------------
        # 4. 平面细化与上采样
        # --------------------------------------------------

        # 三个平面使用相同的输出网络，分别进行计算。
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

        # 输出为潜在特征，保留正负数值，不使用 Sigmoid。
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

        # --------------------------------------------------
        # Step 1：图像特征适配
        # [B, C, H, W] -> [B, D, S, S]
        # --------------------------------------------------
        projected = self.image_projection(
            self.image_pool(image_features)
        )

        # --------------------------------------------------
        # Step 2：构造带位置编码的图像 memory
        # [B, D, S, S] -> [B, S*S, D]
        # --------------------------------------------------
        image_tokens = projected.flatten(2).transpose(1, 2)

        image_pos = self.image_position(
            self.image_coordinates.to(image_tokens)
        ).unsqueeze(0)

        memory = self.memory_norm(
            image_tokens + image_pos
        )

        # --------------------------------------------------
        # Step 3：构造图片条件化的三平面查询
        # --------------------------------------------------

        # 平面内部位置：[1, 1, L*L, D]
        position = self.query_position(
            self.plane_coordinates.to(image_tokens)
        ).reshape(
            1,
            1,
            tokens_per_plane,
            self.inner_dim,
        )

        # 平面身份：[1, 3, 1, D]
        identity = self.plane_embedding.weight.reshape(
            1,
            3,
            1,
            self.inner_dim,
        )

        # 图片条件：[B, 1, 1, D]
        condition = self.image_condition(
            projected.mean(dim=(2, 3))
        ).reshape(
            batch_size,
            1,
            1,
            self.inner_dim,
        )

        # 广播相加，得到 [B, 3, L*L, D]。
        queries = position + identity + condition

        queries = queries.reshape(
            batch_size,
            3 * tokens_per_plane,
            self.inner_dim,
        )

        queries = self.query_norm(queries)

        # --------------------------------------------------
        # Step 4：自注意力、交叉注意力、前馈更新
        # --------------------------------------------------
        for layer in self.decoder_layers:
            queries = layer(
                tgt=queries,
                memory=memory,
            )

        queries = self.output_norm(queries)

        # --------------------------------------------------
        # Step 5：将序列还原为三个二维特征平面
        # --------------------------------------------------

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

        # 将三个平面作为独立样本送入共享输出网络。
        # [B, 3, D, L, L] -> [B*3, D, L, L]
        planes = planes.reshape(
            batch_size * 3,
            self.inner_dim,
            low_res,
            low_res,
        )

        # --------------------------------------------------
        # Step 6：空间细化、上采样和通道转换
        # --------------------------------------------------
        planes = self.plane_readout(planes)

        return planes.reshape(
            batch_size,
            3,
            self.triplane_dim,
            self.triplane_high_res,
            self.triplane_high_res,
        ).contiguous()