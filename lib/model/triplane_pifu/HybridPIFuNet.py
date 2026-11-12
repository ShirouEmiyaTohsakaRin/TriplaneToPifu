import torch
import torch.nn as nn

from ..BasePIFuNet import BasePIFuNet
from ..HGFilters import HGFilter
from ..DepthNormalizer import DepthNormalizer
from ...net_util import init_net

from .TriplaneGenerator import TriplaneGenerator
from .TriplaneSampler import TriplaneSampler
from .HybridOccupancyDecoder import HybridOccupancyDecoder


class HybridPIFuNet(BasePIFuNet):
    """
    在 PIFu 中加入三平面特征，预测三维点位于物体内部的概率。
    支持单张或多张图片，以及正交和透视投影。

    B 是物体数，V 是每个物体的视图数，N 是查询点数。
    图像、相机矩阵和查询点按 B*V 排列，同一物体的视图连续存放。
    每个视图各生成一组三平面；MLP 合并各视图的信息后，输出 B 个物体的结果。
    平面使用图像坐标 u、v 和归一化深度，不直接使用世界坐标 XYZ。

    s(X | I) = f_theta(F_pixel(pi(X)), F_tri(X), z(X))

    filter(images): 提取图像特征、生成三平面，并保存下来供查询使用。
    query(points, calibs): 使用已保存的特征，预测指定点的内部概率（Occupancy）。
    forward(...): 依次执行上面两步；提供真实标签时，还会计算损失。
    """

    def __init__(self, opt, projection_mode="orthogonal", error_term=None):
        if getattr(opt, "num_views", 1) < 1:
            raise ValueError("num_views must be positive.")

        # 先根据数据确定深度范围的中心和半宽，才能把点放到三平面中的正确位置。
        for name in ("triplane_depth_center", "triplane_depth_radius"):
            if getattr(opt, name, None) is None:
                raise ValueError(f"Please configure opt.{name}.")

        if opt.loadSize < 2 or opt.z_size <= 0:
            raise ValueError("loadSize must be >= 2 and z_size must be > 0.")

        super().__init__(
            projection_mode=projection_mode,
            error_term=nn.MSELoss() if error_term is None else error_term,
        )

        self.name = "hybrid_pifu"
        self.opt = opt
        self.num_views = getattr(opt, "num_views", 1)
        self.skip_hourglass = getattr(opt, "skip_hourglass", False)

        # --------------------------------------------------
        # 1. 沿用 PIFu：提取图像特征，并换算深度值
        # --------------------------------------------------
        self.image_filter = HGFilter(opt)
        self.normalizer = DepthNormalizer(opt)

        plane_dim = getattr(opt, "triplane_dim", 32)

        # --------------------------------------------------
        # 2. 用 Transformer 根据图像特征生成三个特征平面
        # --------------------------------------------------
        self.triplane_generator = TriplaneGenerator(
            image_feat_dim=opt.hourglass_dim,
            inner_dim=getattr(opt, "triplane_inner_dim", 256),
            image_token_res=getattr(opt, "triplane_image_token_res", 16),
            triplane_low_res=getattr(opt, "triplane_low_res", 16),
            triplane_high_res=getattr(opt, "triplane_high_res", 64),
            triplane_dim=plane_dim,
            num_layers=getattr(opt, "triplane_num_layers", 4),
            num_heads=getattr(opt, "triplane_num_heads", 8),
            dropout=getattr(opt, "triplane_dropout", 0.0),
        )

        # --------------------------------------------------
        # 3. 换算查询点的坐标，并从三个平面取特征
        # --------------------------------------------------
        self.triplane_sampler = TriplaneSampler(
            depth_center=opt.triplane_depth_center,
            depth_radius=opt.triplane_depth_radius,
        )

        # 启用 skip_hourglass 时，把网络前面提取的浅层特征也拼入图像特征。
        pixel_dim = opt.hourglass_dim
        if self.skip_hourglass:
            pixel_dim += self.image_filter.conv1.out_channels

        # MLP 中间各层的大小沿用 PIFu 配置。
        # 输入特征一共有多少维，由 HybridOccupancyDecoder 根据三类特征自动计算。
        mlp_dim = getattr(
            opt,
            "mlp_dim",
            [257, 1024, 512, 256, 128, 1],
        )

        if len(mlp_dim) < 3 or mlp_dim[-1] != 1:
            raise ValueError(
                "mlp_dim must include hidden layers and end in 1."
            )

        # --------------------------------------------------
        # 4. 合并图像、三平面和深度特征，预测点位于物体内部的概率
        # --------------------------------------------------
        self.hybrid_decoder = HybridOccupancyDecoder(
            pixel_dim=pixel_dim,
            plane_dim=plane_dim,
            hidden_dims=tuple(mlp_dim[1:-1]),
            no_residual=getattr(opt, "no_residual", False),
            num_views=self.num_views,
        )

        # 图像编码器和 MLP 使用 PIFu 的参数初始化方式。
        # 只初始化这两个模块，保留 Transformer 自己的初始参数。
        init_net(self.image_filter)
        init_net(self.hybrid_decoder)

        # 保存 filter() 算出的特征，之后查询不同的点时可以重复使用。
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.triplanes = None

        # 清空上一次查询的预测、标签和范围标记。
        self._clear_query_cache()

    def _clear_query_cache(self):
        self.preds = None
        self.raw_preds = None
        self.labels = None
        self.intermediate_preds_list = []
        self.in_img_mask = None
        self.triplane_valid_mask = None

    def filter(self, images):
        """
        images: [B*V, 3, H, W]

        提取并保存图像特征和三平面。
        同一批图像只需处理一次，之后可多次调用 query() 查询不同的点。
        """
        # 换了一批图像，先清空之前保存的特征和预测。
        self.im_feat_list = []
        self.triplanes = None
        self.tmpx = None
        self.normx = None
        self._clear_query_cache()

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W].")
        if images.shape[0] == 0 or images.shape[0] % self.num_views:
            raise ValueError("Image batch must be a nonempty B*num_views batch.")

        features, self.tmpx, self.normx = self.image_filter(images)

        if not features:
            raise RuntimeError("HGFilter returned no feature maps.")

        # 训练时保留每一级图像特征，各级都做预测并计算损失。
        # 预测时只使用最后一级图像特征。
        self.im_feat_list = (
            features if self.training else [features[-1]]
        )

        # 用最后一级图像特征生成三平面，前面各级做预测时也使用它。
        # 生成的三平面不能调用 detach，否则生成器收不到损失传回的梯度，无法学习。
        self.triplanes = self.triplane_generator(features[-1])

    def _project_points(self, points, calibs, transforms=None):
        """
        用相机矩阵把三维点转换为图像位置 u、v 和深度 d。
        输出形状为 [B*V, 3, N]，三个通道依次是 u、v、d。

        transforms 可为：
            [2, 3]：所有视图使用同一个二维变换矩阵。
            [B*V, 2, 3]：每个视图使用自己的二维变换矩阵。

        这个额外变换只调整归一化后的图像位置，不改变深度。
        """
        batch_size = points.shape[0]

        if (
            calibs.ndim != 3
            or calibs.shape[0] != batch_size
            or calibs.shape[1] not in (3, 4)
            or calibs.shape[2] != 4
        ):
            raise ValueError(
                "calibs must have shape [B, 3, 4] or [B, 4, 4]."
            )

        if calibs.device != points.device:
            raise ValueError(
                "calibs and points must be on the same device."
            )

        # 按配置选择正交或透视投影，得到图像位置和深度。
        # 额外的二维图像变换在下面单独处理。
        xyz = self.projection(
            points,
            calibs.to(dtype=points.dtype),
        )

        if transforms is not None:
            if transforms.device != points.device:
                raise ValueError(
                    "transforms and points must share a device."
                )

            if transforms.ndim == 2:
                if transforms.shape != (2, 3):
                    raise ValueError(
                        "Shared transforms must have shape [2, 3]."
                    )

                transforms = transforms.unsqueeze(0).expand(
                    batch_size, -1, -1
                )

            if transforms.shape != (batch_size, 2, 3):
                raise ValueError(
                    "Batched transforms must have shape [B, 2, 3]."
                )

            transforms = transforms.to(dtype=xyz.dtype)

            uv = torch.bmm(
                transforms[:, :, :2],
                xyz[:, :2, :],
            )
            uv = uv + transforms[:, :, 2:3]

            xyz = torch.cat(
                [uv, xyz[:, 2:3, :]],
                dim=1,
            )

        return xyz

    def query(self, points, calibs, transforms=None, labels=None):
        """
        points:
            [B*V, 3, N]，世界坐标中的三维点；同一物体的点按视图数复制，顺序保持一致。

        calibs:
            [B*V, 3, 4] 或 [B*V, 4, 4]，每个视图对应一个相机矩阵。

        labels:
            可选的真实标签 [B, 1, N]，物体内部为 1，外部为 0。

        返回并保存：
            [B, 1, N]，每个点位于物体内部的预测概率。
        """
        self._clear_query_cache()

        if self.triplanes is None or not self.im_feat_list:
            raise RuntimeError(
                "Call filter(images) before query()."
            )

        if (
            points.ndim != 3
            or points.shape[1] != 3
            or points.shape[0] == 0
            or points.shape[2] == 0
            or not points.is_floating_point()
        ):
            raise ValueError(
                "points must be a nonempty floating [B, 3, N] tensor."
            )

        if points.shape[0] != self.triplanes.shape[0]:
            raise ValueError(
                "points and cached image features have different batches."
            )

        if points.device != self.triplanes.device:
            raise ValueError(
                "points and image features must share a device."
            )

        if labels is not None:
            if labels.shape != (points.shape[0] // self.num_views, 1, points.shape[2]):
                raise ValueError(
                    "labels must have shape [B, 1, N]."
                )

            if labels.device != points.device:
                raise ValueError(
                    "labels and points must share a device."
                )

        # 没有传入标签时也要设为 None，避免误用上一次查询的标签。
        self.labels = labels

        # --------------------------------------------------
        # 步骤 1：先计算投影坐标，图像分支和三平面分支都用这份坐标
        # --------------------------------------------------
        xyz = self._project_points(
            points,
            calibs,
            transforms,
        )

        xy = xyz[:, :2, :]
        depth = xyz[:, 2:3, :]

        # --------------------------------------------------
        # 步骤 2：换算深度，再从三个平面取出这个点的特征
        # --------------------------------------------------
        # 换算前会检查坐标，遇到 NaN 或无穷大时直接报错。
        q = self.triplane_sampler.normalize_coordinates(xyz)

        tri_feat, tri_valid = self.triplane_sampler(
            self.triplanes,
            q,
            return_mask=True,
        )

        self.triplane_valid_mask = tri_valid

        # --------------------------------------------------
        # 步骤 3：标记点的投影是否落在图像范围内
        # --------------------------------------------------
        # 这里只检查投影位置是否在图像边界内，
        # 不判断该像素是否属于人体，也不判断是否被遮挡。
        self.in_img_mask = (
            (xy >= -1.0) & (xy <= 1.0)
        ).all(dim=1, keepdim=True)
        if self.projection_mode == "perspective":
            self.in_img_mask = self.in_img_mask & (depth > 1e-6)
        object_mask = self.in_img_mask.reshape(
            -1, self.num_views, 1, points.shape[2]
        ).any(dim=1)

        # --------------------------------------------------
        # 步骤 4：按 PIFu 原来的方式换算深度，供 MLP 使用
        # --------------------------------------------------
        # 这个深度值与三平面使用的深度坐标换算方式不同，不能直接用 q 的第三维替代。
        depth_feat = self.normalizer(
            depth,
            calibs=calibs,
        )

        # 启用 skip_hourglass 时，也从浅层特征图取值。
        tmp_feat = None
        if self.skip_hourglass:
            tmp_feat = self.index(
                self.tmpx,
                xy.to(dtype=self.tmpx.dtype),
            )

        # --------------------------------------------------
        # 步骤 5：每一级图像特征都与同一份三平面特征、深度一起做预测
        # --------------------------------------------------
        for im_feat in self.im_feat_list:
            pixel_feat = self.index(
                im_feat,
                xy.to(dtype=im_feat.dtype),
            )

            if tmp_feat is not None:
                pixel_feat = torch.cat(
                    [pixel_feat, tmp_feat],
                    dim=1,
                )

            # 将三类特征交给 MLP；多视图时，它会合并图像范围内的视图信息。
            raw_pred = self.hybrid_decoder(
                pixel_feat,
                tri_feat,
                depth_feat,
                view_mask=self.in_img_mask,
            )

            # 点在所有视图的图像范围外时，预测设为 0。
            # 仅深度超出三平面范围时仍保留预测，不能直接认定点在物体外部。
            pred = raw_pred.masked_fill(
                ~object_mask,
                0.0,
            )

            self.intermediate_preds_list.append(pred)
            self.raw_preds = raw_pred

        self.preds = self.intermediate_preds_list[-1]

        return self.preds

    def get_im_feat(self):
        """返回最后一级图像特征，供原有 PIFu 接口使用。"""
        if not self.im_feat_list:
            raise RuntimeError("Call filter(images) first.")

        return self.im_feat_list[-1]

    def get_error(self):
        """分别计算各级预测与真实标签的误差，再取平均作为训练损失。"""
        if self.labels is None:
            raise RuntimeError(
                "Labels are required to compute the loss."
            )

        if not self.intermediate_preds_list:
            raise RuntimeError(
                "Call query() before get_error()."
            )

        losses = [
            self.error_term(
                pred,
                self.labels.to(dtype=pred.dtype),
            )
            for pred in self.intermediate_preds_list
        ]

        return torch.stack(losses).mean()

    def forward(
        self,
        images,
        points,
        calibs,
        transforms=None,
        labels=None,
    ):
        """
        提取图像特征，再预测各查询点位于物体内部的概率。
        提供真实标签 labels 时返回 (occupancy, loss)，否则返回 (occupancy, None)。
        """
        self.filter(images)

        self.query(
            points,
            calibs,
            transforms=transforms,
            labels=labels,
        )

        error = (
            self.get_error()
            if labels is not None
            else None
        )

        return self.get_preds(), error
