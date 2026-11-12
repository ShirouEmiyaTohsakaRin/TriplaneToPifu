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
    单/多视图 PIFu + tri-plane，支持正交与透视投影。
    图像、标定、查询点沿视图展平为 B*V；标签和输出保留 B。
    每个视图生成一组三平面，查询同一世界点后在 MLP 中聚合。
    透视模式下平面参数为 (u, v, normalized depth)，即视锥参数化。

    s(X | I) = f_theta(F_pixel(pi(X)), F_tri(X), z(X))

    filter(images): 编码图像，生成并缓存三平面。
    query(points, calibs): 查询 Occupancy，复用 filter 的缓存。
    forward(...): filter + query，可同时返回监督损失。
    """

    def __init__(self, opt, projection_mode="orthogonal", error_term=None):
        if getattr(opt, "num_views", 1) < 1:
            raise ValueError("num_views must be positive.")

        # 这两个量必须由数据的投影深度范围确定，不能随意默认。
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
        # 1. PIFu 图像编码器和深度归一化器
        # --------------------------------------------------
        self.image_filter = HGFilter(opt)
        self.normalizer = DepthNormalizer(opt)

        plane_dim = getattr(opt, "triplane_dim", 32)

        # --------------------------------------------------
        # 2. 新版 Transformer 三平面生成器
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
        # 3. 三平面坐标归一化和特征采样
        # --------------------------------------------------
        self.triplane_sampler = TriplaneSampler(
            depth_center=opt.triplane_depth_center,
            depth_radius=opt.triplane_depth_radius,
        )

        # 可选浅层特征也属于 F_pixel，不增加第四类函数输入。
        pixel_dim = opt.hourglass_dim
        if self.skip_hourglass:
            pixel_dim += self.image_filter.conv1.out_channels

        # 沿用原 PIFu 的隐藏层配置；
        # 输入维度由 HybridOccupancyDecoder 自动计算。
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
        # 4. 三输入 Occupancy 解码器
        # --------------------------------------------------
        self.hybrid_decoder = HybridOccupancyDecoder(
            pixel_dim=pixel_dim,
            plane_dim=plane_dim,
            hidden_dims=tuple(mlp_dim[1:-1]),
            no_residual=getattr(opt, "no_residual", False),
            num_views=self.num_views,
        )

        # 只对 PIFu 部分使用原有初始化。
        # 不对整个 self 调用 init_net，以保留 Transformer 自身的初始化。
        init_net(self.image_filter)
        init_net(self.hybrid_decoder)

        # 图像级缓存：由 filter() 更新。
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.triplanes = None

        # 查询级缓存：由 query() 更新。
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

        编码图像并生成三平面。
        每批图像只需调用一次，随后可以多次 query()。
        """
        # 清除上一批图像的缓存，避免误用旧查询结果。
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

        # 训练：保留每一级图像特征，用于中间监督。
        # 推理：只使用最后一级图像特征。
        self.im_feat_list = (
            features if self.training else [features[-1]]
        )

        # 所有 Hourglass 预测分支共享这一组三平面。
        # 这里不能 detach，否则 Generator 无法从 Occupancy 损失学习。
        self.triplanes = self.triplane_generator(features[-1])

    def _project_points(self, points, calibs, transforms=None):
        """
        输出 [B, 3, N] = (u, v, d)。

        transforms 可为：
            [2, 3]：整个批次共享的二维仿射变换。
            [B, 2, 3]：每个样本独立的二维仿射变换。

        变换作用于归一化图像坐标，不改变深度。
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

        # 使用选择的投影；
        # 二维变换在下面显式按批次处理。
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
            [B*V, 3, N]，同一物体的世界坐标按视图重复。

        calibs:
            [B*V, 3, 4] 或 [B*V, 4, 4]。

        labels:
            可选 [B, 1, N]，内部为 1，外部为 0。

        返回并缓存：
            [B, 1, N] Occupancy。
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

        # 每次查询都更新，包括 labels=None，防止沿用旧标签。
        self.labels = labels

        # --------------------------------------------------
        # Step 1：统一投影，保证两个采样分支坐标对齐
        # --------------------------------------------------
        xyz = self._project_points(
            points,
            calibs,
            transforms,
        )

        xy = xyz[:, :2, :]
        depth = xyz[:, 2:3, :]

        # --------------------------------------------------
        # Step 2：三平面特征
        # --------------------------------------------------
        # Sampler 同时检查坐标是否有限。
        q = self.triplane_sampler.normalize_coordinates(xyz)

        tri_feat, tri_valid = self.triplane_sampler(
            self.triplanes,
            q,
            return_mask=True,
        )

        self.triplane_valid_mask = tri_valid

        # --------------------------------------------------
        # Step 3：图像范围掩码
        # --------------------------------------------------
        # 仅判断是否位于图像方形采样范围内，
        # 不是人体前景掩码。
        self.in_img_mask = (
            (xy >= -1.0) & (xy <= 1.0)
        ).all(dim=1, keepdim=True)
        if self.projection_mode == "perspective":
            self.in_img_mask = self.in_img_mask & (depth > 1e-6)
        object_mask = self.in_img_mask.reshape(
            -1, self.num_views, 1, points.shape[2]
        ).any(dim=1)

        # --------------------------------------------------
        # Step 4：PIFu 深度特征
        # --------------------------------------------------
        # 不使用 q 的第三维代替。
        depth_feat = self.normalizer(
            depth,
            calibs=calibs,
        )

        # 可选的浅层像素特征。
        tmp_feat = None
        if self.skip_hourglass:
            tmp_feat = self.index(
                self.tmpx,
                xy.to(dtype=self.tmpx.dtype),
            )

        # --------------------------------------------------
        # Step 5：各级图像特征分别与共享三平面特征融合
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

            # 三输入核心函数。
            raw_pred = self.hybrid_decoder(
                pixel_feat,
                tri_feat,
                depth_feat,
                view_mask=self.in_img_mask,
            )

            # 延续 PIFu：图像范围外的点输出 0。
            #
            # 不乘 tri_valid：
            # 深度超出平面范围不等同于物体外部。
            pred = raw_pred.masked_fill(
                ~object_mask,
                0.0,
            )

            self.intermediate_preds_list.append(pred)
            self.raw_preds = raw_pred

        self.preds = self.intermediate_preds_list[-1]

        return self.preds

    def get_im_feat(self):
        """兼容 PIFu：返回最后一级图像特征。"""
        if not self.im_feat_list:
            raise RuntimeError("Call filter(images) first.")

        return self.im_feat_list[-1]

    def get_error(self):
        """各级 Hourglass 的 Occupancy 损失取平均。"""
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
        有 labels：返回 (occupancy, loss)。
        无 labels：返回 (occupancy, None)。
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
