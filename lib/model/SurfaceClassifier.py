import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature, view_mask=None):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        if self.num_views < 1 or feature.shape[0] % self.num_views:
            raise ValueError('Feature batch must be divisible by num_views.')
        if view_mask is not None and view_mask.shape != (feature.shape[0], 1, feature.shape[2]):
            raise ValueError('view_mask must have shape [B*V, 1, N].')

        def aggregate(value):
            value = value.reshape(-1, self.num_views, value.shape[1], value.shape[2])
            if view_mask is None:
                return value.mean(dim=1)
            mask = view_mask.reshape(-1, self.num_views, 1, value.shape[-1]).to(value.dtype)
            return (value * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = aggregate(y)
                tmpy = aggregate(feature)

        if self.last_op:
            y = self.last_op(y)

        return y
