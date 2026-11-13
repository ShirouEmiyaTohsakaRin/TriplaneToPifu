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
    目的：把 PIFu 和三平面接在一起，判断三维点是否在物体内部。
    每个点同时参考图像特征、三平面特征和深度，让网络学会如何结合这三种依据。
    特征就是网络提取的一组数；三平面让不同深度的点有机会取到不同的特征。

    filter(images)：先处理图像，保存后面要用的特征和三平面。
    query(points, calibs)：再预测指定点位于物体内部的概率。
    forward()：连续完成这两步，传入真实标签时还会计算预测误差。
    核心公式：s(X | I) = f_theta(F_pixel(pi(X)), F_tri(X), z(X))。

    B 是物体数，V 是每个物体的图像数，N 是查询点数。
    输入按 B*V 排列，同一物体的图像、相机矩阵和点要对应好。
    每张图生成自己的三个平面，网络合并各视图后输出 [B, 1, N]。
    支持正交和透视相机；平面坐标使用图像位置 u、v 和换算后的深度。
    """

    def __init__(self, opt, projection_mode="orthogonal", error_term=None):
        if getattr(opt, "num_views", 1) < 1:
            raise ValueError("num_views must be positive.")

        # 先确定数据中的深度范围，后面才能算出点在三平面上的位置。
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

        # 1. 沿用 PIFu，从图像提取特征，再用深度区分同一像素前后的点。
        self.image_filter = HGFilter(opt)
        self.normalizer = DepthNormalizer(opt)

        plane_dim = getattr(opt, "triplane_dim", 32)

        # 2. 增加三平面，让不同深度的点可以取到不同的特征。
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

        # 3. 用采样器把三维点对应到三个平面，取出该点的特征。
        self.triplane_sampler = TriplaneSampler(
            depth_center=opt.triplane_depth_center,
            depth_radius=opt.triplane_depth_radius,
        )

        # 开启 skip_hourglass 后，预测时还会参考网络第一层提取的特征。
        # pixel_dim 要加上这份特征的长度，否则输入大小对不上。
        pixel_dim = opt.hourglass_dim
        if self.skip_hourglass:
            pixel_dim += self.image_filter.conv1.out_channels

        # 预测网络中间各层的大小沿用 PIFu 配置。
        # 输入大小由解码器根据图像、三平面和深度三部分自动计算。
        mlp_dim = getattr(
            opt,
            "mlp_dim",
            [257, 1024, 512, 256, 128, 1],
        )

        if len(mlp_dim) < 3 or mlp_dim[-1] != 1:
            raise ValueError(
                "mlp_dim must include hidden layers and end in 1."
            )

        # 4. 把图像特征、三平面特征和深度交给预测网络，判断点是否在内部。
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

        # 清空上一次查询留下的结果。
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
        目的：把图像先算一次，保存特征和三平面，后面查询不同点时可以重复使用。
        这些结果只取决于图像，不必每查一个点就重新生成。

        images：[B*V, 3, H, W]，同一物体的各张图像放在一起。
        换了图像，就要重新调用本函数。
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

        # 训练时各阶段都预测并计算误差，让每个阶段都能学习。
        # 实际重建只用最后一个阶段，减少计算。
        self.im_feat_list = (
            features if self.training else [features[-1]]
        )

        # 三平面只用 features[-1] 生成一次，供各阶段的预测共用，避免重复生成。
        # 这里不能对三平面调用 detach，否则预测误差传不回生成器，它就无法学习。
        self.triplanes = self.triplane_generator(features[-1])

    def _project_points(self, points, calibs, transforms=None):
        """
        目的：算出三维点落在图像上的哪里，后面才能在对应位置取特征。
        返回 [B*V, 3, N]，三个通道依次是横坐标 u、纵坐标 v、深度 d。

        transforms 可额外平移或缩放图像位置，不改变深度。
        [2, 3] 表示所有图像使用同一个变换；[B*V, 2, 3] 表示每张图单独变换。
        这里使用换算后的图像坐标，图像边界对应 -1 和 1。
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
        目的：从图像和三个平面取出每个点的特征，再结合深度判断它是否在物体内部。
        三份输入必须属于同一个点，否则网络就会把不同点的依据混在一起。
        调用前先用 filter(images) 处理图像。

        points：[B*V, 3, N]，世界坐标点；同一物体的点按图像数复制，顺序不变。
        calibs：[B*V, 3, 4] 或 [B*V, 4, 4]，每张图对应的相机矩阵。
        labels：可选 [B, 1, N]，真实答案，物体内部填 1，外部填 0。
        返回并保存 [B, 1, N]；越接近 1，网络越倾向于判断点在物体内部。
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

        # 1. 先算出点在图像上的位置 xy 和深度 depth，后面才能找到对应特征。
        xyz = self._project_points(
            points,
            calibs,
            transforms,
        )

        xy = xyz[:, :2, :]
        depth = xyz[:, 2:3, :]

        # 2. 为了区分同一像素前后的点，用图像位置和深度查询三平面，得到 tri_feat。
        # 换算前会检查坐标，遇到 NaN 或无穷大时直接报错。
        q = self.triplane_sampler.normalize_coordinates(xyz)

        tri_feat, tri_valid = self.triplane_sampler(
            self.triplanes,
            q,
            return_mask=True,
        )

        self.triplane_valid_mask = tri_valid

        # 3. 点落在某张图的边界外时，不使用这张图的特征。
        # 这个标记不判断人体轮廓或遮挡；透视相机还要求点的深度为正。
        self.in_img_mask = (
            (xy >= -1.0) & (xy <= 1.0)
        ).all(dim=1, keepdim=True)
        if self.projection_mode == "perspective":
            self.in_img_mask = self.in_img_mask & (depth > 1e-6)
        object_mask = self.in_img_mask.reshape(
            -1, self.num_views, 1, points.shape[2]
        ).any(dim=1)

        # 4. 按 PIFu 原来的比例缩放深度，让网络知道点的前后位置。
        # 它与三平面坐标 q 使用不同的缩放方式，不能直接拿 q 的深度来代替。
        depth_feat = self.normalizer(
            depth,
            calibs=calibs,
        )

        # 开启 skip_hourglass 后，从 tmpx 取出第一层在 xy 处的特征，稍后一起用于预测。
        # tmpx 已做 detach：这里只借用这份特征，误差不会沿这条路径传回第一层。
        tmp_feat = None
        if self.skip_hourglass:
            tmp_feat = self.index(
                self.tmpx,
                xy.to(dtype=self.tmpx.dtype),
            )

        # 5. 各阶段的图像特征分别预测一次，训练时把这些预测误差取平均。
        # 每次都使用同一个预测网络和同一组三平面。
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

            # 把同一点的三份输入放在一起，让网络学会根据它们判断内外。
            raw_pred = self.hybrid_decoder(
                pixel_feat,
                tri_feat,
                depth_feat,
                view_mask=self.in_img_mask,
            )

            # 没有任何视图满足图像范围要求时，结果设为 0。
            # 如果只是深度超出三平面范围，仍保留网络的预测。
            pred = raw_pred.masked_fill(
                ~object_mask,
                0.0,
            )

            self.intermediate_preds_list.append(pred)
            self.raw_preds = raw_pred

        self.preds = self.intermediate_preds_list[-1]

        return self.preds

    def get_im_feat(self):
        """
        目的：让其他模块直接使用最后一张特征图，避免重复处理图像。
        保留这个函数，也能让原有 PIFu 代码继续调用。
        """
        if not self.im_feat_list:
            raise RuntimeError("Call filter(images) first.")

        return self.im_feat_list[-1]

    def get_error(self):
        """
        目的：算出预测错了多少，供训练时调整模型。
        默认将预测值与真实标签的差平方，再取平均。
        Hourglass 的各阶段都会做一次预测；把这些误差平均起来，让各阶段都能学习。
        """
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
        目的：一次完成图像处理、点预测和误差计算，方便训练代码调用。
        有 labels 时返回（预测概率，误差），没有时返回（预测概率，None）。
        这里只计算结果，模型参数由外面的训练代码根据误差更新。
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
