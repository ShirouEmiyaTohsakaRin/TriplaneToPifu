import torch
import torch.nn as nn


def make_grid_coordinates(resolution):
    """
    在 [-1, 1] 范围内均匀放置 resolution × resolution 个网格点。
    返回 [resolution * resolution, 2]，每行依次是水平坐标、垂直坐标。
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
    根据 PIFu 图像特征，生成三个可供三维点查询的特征平面。
    每个平面位置存放一组特征数值，不是图片的 RGB 颜色。

    输入：
        image_features: [B, image_feat_dim, H, W]

    输出：
        planes: [B, 3, triplane_dim, R, R]

    B 是输入图像数；多视图时，同一物体的每张图像分别生成一组三平面。
    R 是输出平面的边长。代码用 xy、xz、yz 表示三个平面：
        0: xy，横轴是图像横坐标 u，纵轴是图像纵坐标 v。
        1: xz，横轴是 u，纵轴是归一化深度。
        2: yz，横轴是 v，纵轴是归一化深度。

    先告诉 Transformer 每个位置在哪、属于哪个平面，以及图片的整体信息；
    再用 PyTorch 的 TransformerDecoderLayer 从图像特征中读取信息。
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

        # 网格坐标是固定的，不参与训练；加载模型时可按配置重建，无需存入权重文件。
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
        # 1. 缩小图像特征图，并把每个网格位置的特征作为一个 token
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
        # 2. 为三个平面的每个网格位置准备初始特征，交给 Transformer 查询图像
        # --------------------------------------------------

        # 把网格坐标转换成特征，让网络知道这是平面上的哪个位置。
        self.query_position = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        # 给 xy、xz、yz 各准备一组可学习的数值，让网络区分三个平面。
        self.plane_embedding = nn.Embedding(
            3,
            inner_dim,
        )

        nn.init.normal_(
            self.plane_embedding.weight,
            std=0.02,
        )

        # 把当前图片的整体信息加入每个平面位置的初始特征。
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
        # 3. 用 Transformer 让平面位置之间交换信息，并读取图像特征
        # --------------------------------------------------

        # 每层单独创建，各自初始化参数。
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
        # 4. 放大特征平面，并用卷积调整每个位置的特征
        # --------------------------------------------------

        # 三个平面分别处理，但共用同一套卷积参数。
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

        # 这里输出的是特征，不是概率，因此保留正负值，不用 Sigmoid 压到 0～1。
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
        # 步骤 1：缩小图像特征图，并调整为 Transformer 使用的特征维度
        # [B, C, H, W] -> [B, D, S, S]
        # --------------------------------------------------
        projected = self.image_projection(
            self.image_pool(image_features)
        )

        # --------------------------------------------------
        # 步骤 2：把图像特征排成序列，加入位置信息，供 Transformer 读取
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
        # 步骤 3：用网格位置、平面编号和图片整体信息，组成平面查询的初始特征
        # --------------------------------------------------

        # 每个网格位置对应的特征：[1, 1, L*L, D]
        position = self.query_position(
            self.plane_coordinates.to(image_tokens)
        ).reshape(
            1,
            1,
            tokens_per_plane,
            self.inner_dim,
        )

        # 区分三个平面的特征：[1, 3, 1, D]
        identity = self.plane_embedding.weight.reshape(
            1,
            3,
            1,
            self.inner_dim,
        )

        # 每张图片的整体特征：[B, 1, 1, D]
        condition = self.image_condition(
            projected.mean(dim=(2, 3))
        ).reshape(
            batch_size,
            1,
            1,
            self.inner_dim,
        )

        # 利用广播把三部分相加，让每张图、每个平面、每个位置都有对应特征。
        # 结果形状：[B, 3, L*L, D]。
        queries = position + identity + condition

        queries = queries.reshape(
            batch_size,
            3 * tokens_per_plane,
            self.inner_dim,
        )

        queries = self.query_norm(queries)

        # --------------------------------------------------
        # 步骤 4：平面位置之间交换信息（自注意力），再读取图像信息（交叉注意力）并更新特征
        # --------------------------------------------------
        for layer in self.decoder_layers:
            queries = layer(
                tgt=queries,
                memory=memory,
            )

        queries = self.output_norm(queries)

        # --------------------------------------------------
        # 步骤 5：把更新后的序列重新排成三个二维网格
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

        # 把三个平面分别送入同一个卷积网络处理。
        # [B, 3, D, L, L] -> [B*3, D, L, L]
        planes = planes.reshape(
            batch_size * 3,
            self.inner_dim,
            low_res,
            low_res,
        )

        # --------------------------------------------------
        # 步骤 6：放大平面，并调整到指定的输出特征维度
        # --------------------------------------------------
        planes = self.plane_readout(planes)

        return planes.reshape(
            batch_size,
            3,
            self.triplane_dim,
            self.triplane_high_res,
            self.triplane_high_res,
        ).contiguous()
