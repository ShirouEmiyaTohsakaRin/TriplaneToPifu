"""Hybrid-only model lifecycle, including legacy state_dict compatibility."""
from pathlib import Path
import math
import random
import numpy as np
import torch

SHAPE_OPTIONS = (
    'loadSize', 'z_size', 'num_views', 'projection_mode', 'norm', 'num_stack',
    'num_hourglass', 'skip_hourglass', 'hg_down', 'hourglass_dim', 'mlp_dim',
    'no_residual', 'triplane_dim', 'triplane_inner_dim', 'triplane_image_token_res',
    'triplane_low_res', 'triplane_high_res', 'triplane_num_layers',
    'triplane_num_heads', 'triplane_dropout', 'triplane_depth_center',
    'triplane_depth_radius',
)
COLOR_OPTIONS = (
    'loadSize', 'z_size', 'num_views', 'projection_mode', 'hourglass_dim',
    'norm_color', 'mlp_dim_color', 'no_residual', 'use_tanh', 'color_loss_type',
)


def device_for(opt):
    return torch.device('cuda:%d' % opt.gpu_id if opt.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_checkpoint(path):
    if path is None:
        return None
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, dict):
        raise ValueError('Checkpoint must contain a state_dict.')
    return value if 'state_dict' in value else {'state_dict': value, 'options': {}}


def restore_options(opt, checkpoint, kind='shape'):
    """Restore architecture before constructing data/model; never silently override CLI."""
    if checkpoint is None:
        return
    if checkpoint.get('kind', kind) != kind:
        raise ValueError('Wrong checkpoint kind: expected ' + kind)
    saved = dict(checkpoint.get('options', {}))
    if kind == 'shape':
        state = checkpoint['state_dict']
        if not any(k.startswith('triplane_generator.') for k in state):
            raise ValueError('Original PIFu weights are initialization only. Use --init_pifu_checkpoint during shape training.')
        for field in ('depth_center', 'depth_radius'):
            key = 'triplane_sampler.' + field
            if key not in state:
                raise ValueError('Hybrid checkpoint is missing ' + key)
            saved['triplane_' + field] = float(state[key])
    explicit = set(getattr(opt, '_explicit_options', []))
    for key in SHAPE_OPTIONS if kind == 'shape' else COLOR_OPTIONS:
        if key not in saved:
            continue
        if kind == 'color' and key == 'num_views':
            continue  # Geometry determines grouping for both networks.
        if kind == 'color' and key in ('loadSize', 'z_size', 'projection_mode', 'hourglass_dim', 'no_residual'):
            current = getattr(opt, key)
            if current is not None and current != saved[key]:
                raise ValueError(f'Color and geometry checkpoints disagree on {key}.')
        if key in explicit:
            # A trained shared network can be queried with a different number of views.
            if key == 'num_views':
                continue
            requested = getattr(opt, key)
            same = (math.isclose(requested, saved[key], rel_tol=1e-5, abs_tol=1e-7)
                    if isinstance(requested, float) and isinstance(saved[key], (float, int))
                    else requested == saved[key])
            if not same:
                raise ValueError(f'{key} conflicts with checkpoint: {requested!r} != {saved[key]!r}')
        setattr(opt, key, saved[key])


def save_checkpoint(path, model, opt, kind='shape', optimizer=None, epoch=0, epoch_complete=False):
    keys = SHAPE_OPTIONS if kind == 'shape' else COLOR_OPTIONS
    payload = {
        'format_version': 1, 'kind': kind, 'state_dict': model.state_dict(),
        'options': {key: getattr(opt, key) for key in keys},
        'epoch': epoch, 'epoch_complete': epoch_complete,
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def resume_optimizer(optimizer, checkpoint, opt):
    if not opt.continue_train:
        return 0
    if checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'] + int(checkpoint.get('epoch_complete', False))
    return max(opt.resume_epoch, 0)


def resume_path(opt, prefix):
    suffix = 'latest' if opt.resume_epoch < 0 else 'epoch_%d' % opt.resume_epoch
    return str(Path(opt.checkpoints_path) / opt.name / (prefix + '_' + suffix))


def initialize_pifu_encoder(model, path):
    state = read_checkpoint(path)['state_dict']
    encoder = {key[len('image_filter.'):]: value for key, value in state.items()
               if key.startswith('image_filter.')}
    if not encoder:
        raise ValueError('No PIFu image_filter weights found.')
    model.image_filter.load_state_dict(encoder, strict=True)
    print('Initialized PIFu image encoder; triplane and hybrid decoder still require training.')


def set_projection(opt, dataset_mode):
    if opt.projection_mode is not None and opt.projection_mode != dataset_mode:
        raise ValueError(f'Data uses {dataset_mode}, model requests {opt.projection_mode}. '
                         'Use a calibrated manifest for perspective data.')
    opt.projection_mode = dataset_mode


def configure_depth(opt, dataset):
    """Compute a conservative projected-depth interval from training data/calibration."""
    center, radius = opt.triplane_depth_center, opt.triplane_depth_radius
    if center is not None and radius is not None:
        return
    if (center is None) != (radius is None):
        raise ValueError('Supply both triplane depth center and radius, or neither.')
    if hasattr(dataset, 'depth_bounds'):
        low, high = dataset.depth_bounds()
    else:
        low, high = float('inf'), -float('inf')
        for subject in dataset.subjects:
            bounds = dataset.mesh_dic[subject].bounds
            lower = np.minimum(bounds[0] - 3 * opt.sigma, dataset.B_MIN)
            upper = np.maximum(bounds[1] + 3 * opt.sigma, dataset.B_MAX)
            for path in sorted((Path(dataset.PARAM) / subject).glob('*.npy')):
                param = np.load(path, allow_pickle=True).item()
                row = (param['scale'] / param['ortho_ratio'] / (opt.loadSize // 2)) * np.asarray(param['R'])[2]
                offset = -row @ np.asarray(param['center'])
                lo = np.minimum(row * lower, row * upper).sum() + offset
                hi = np.maximum(row * lower, row * upper).sum() + offset
                factors = (0.9, 1.1) if opt.random_scale else (1.,)
                values = [v * factor for factor in factors for v in (lo, hi)]
                low, high = min(low, *values), max(high, *values)
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('Cannot estimate projected depth; specify center/radius explicitly.')
    opt.triplane_depth_center = float((low + high) / 2)
    opt.triplane_depth_radius = float((high - low) / 2 * 1.05)
    print('Estimated triplane depth center/radius:', opt.triplane_depth_center, opt.triplane_depth_radius)
