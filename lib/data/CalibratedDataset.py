"""Explicit camera/image groups for multi-view inference and supervised training."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CalibratedDataset(Dataset):
    def __init__(self, opt, path, phase='eval'):
        self.opt = opt
        self.root = Path(path).resolve().parent
        document = json.loads(Path(path).read_text())
        self.projection_mode = document['projection_mode']
        if self.projection_mode not in ('orthogonal', 'perspective'):
            raise ValueError('Manifest projection_mode must be orthogonal or perspective.')
        self.records = [record for record in document['subjects']
                        if phase == 'eval' or record.get('split', phase) == phase]
        if not self.records:
            raise ValueError('No subjects for manifest phase ' + phase)
        self.phase = phase
        self.is_train = phase == 'train'
        if self.is_train and any((opt.random_flip, opt.random_trans, opt.random_scale)):
            raise ValueError('Calibrated manifests contain final normalized cameras; geometric augmentation is not implemented for this loader.')
        self.to_image = transforms.Compose([
            transforms.Resize((opt.loadSize, opt.loadSize)), transforms.ToTensor(),
            transforms.Normalize((.5,) * 3, (.5,) * 3),
        ])
        self.to_mask = transforms.Compose([
            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        for record in self.records:
            if len(record['views']) < opt.num_views or (phase == 'eval' and len(record['views']) != opt.num_views):
                raise ValueError(f"{record['name']}: view count does not match num_views={opt.num_views}")
            if Path(record['name']).name != record['name'] or record['name'] in ('.', '..'):
                raise ValueError('Subject name must be a plain filename.')
            bounds = np.asarray([record['b_min'], record['b_max']], dtype=np.float32)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
                raise ValueError('Require finite b_min/b_max with positive extent.')

    def __len__(self):
        return len(self.records)

    def calibration(self, view):
        value = view['calib']
        matrix = np.load(self.root / value, allow_pickle=False) if isinstance(value, str) else np.asarray(value)
        if matrix.shape == (3, 4):
            matrix = np.vstack([matrix, [0, 0, 0, 1]])
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('calib must be a finite 3x4 or 4x4 matrix.')
        return torch.as_tensor(matrix, dtype=torch.float32)

    def _supervision(self, record):
        if 'samples' not in record:
            raise ValueError('Training manifest requires a samples NPZ for ' + record['name'])
        with np.load(self.root / record['samples'], allow_pickle=False) as archive:
            result = {}
            for point_key, label_key, count, channels in (
                ('samples', 'labels', self.opt.num_sample_inout, 1),
                ('color_samples', 'rgbs', self.opt.num_sample_color, 3),
            ):
                if not count:
                    continue
                points, labels = archive[point_key], archive[label_key]
                if points.ndim != 2 or points.shape[0] != 3 or points.shape[1] < 1 or labels.shape != (channels, points.shape[1]):
                    raise ValueError(f'{point_key}/{label_key}: expected [3,N]/[{channels},N].')
                if not np.isfinite(points).all() or not np.isfinite(labels).all():
                    raise ValueError('Nonfinite supervision.')
                low = 0 if label_key == 'labels' else -1
                if labels.min() < low or labels.max() > 1:
                    raise ValueError(f'{label_key} must be in [{low},1].')
                rng = np.random if self.is_train else np.random.RandomState(1991)
                indices = rng.choice(points.shape[1], count, replace=points.shape[1] < count)
                result[point_key] = torch.from_numpy(points[:, indices].astype(np.float32))
                result[label_key] = torch.from_numpy(labels[:, indices].astype(np.float32))
            return result

    def __getitem__(self, index):
        record = self.records[index]
        views = record['views']
        if self.is_train and self.opt.random_multiview:
            indices = np.random.choice(len(views), self.opt.num_views, replace=False)
        else:
            indices = np.linspace(0, len(views), self.opt.num_views, endpoint=False, dtype=int)
        images, masks, calibs = [], [], []
        for index in indices:
            view = views[index]
            with Image.open(self.root / view['image']) as image:
                image = self.to_image(image.convert('RGB'))
            with Image.open(self.root / view['mask']) as mask:
                mask = self.to_mask(mask.convert('L'))
            images.append(image * mask)
            masks.append(mask)
            calibs.append(self.calibration(view))
        result = dict(name=record['name'], img=torch.stack(images), mask=torch.stack(masks),
                      calib=torch.stack(calibs), b_min=np.asarray(record['b_min']), b_max=np.asarray(record['b_max']))
        if self.phase != 'eval':
            result.update(self._supervision(record))
        return result

    def depth_bounds(self):
        low, high = float('inf'), -float('inf')
        for record in self.records:
            lower, upper = np.asarray(record['b_min']), np.asarray(record['b_max'])
            with np.load(self.root / record['samples'], allow_pickle=False) as archive:
                for key in ('samples', 'color_samples'):
                    if key in archive:
                        lower = np.minimum(lower, archive[key].min(axis=1))
                        upper = np.maximum(upper, archive[key].max(axis=1))
            for view in record['views']:
                matrix = self.calibration(view).numpy()
                row = matrix[2, :3]
                low = min(low, np.minimum(row * lower, row * upper).sum() + matrix[2, 3])
                high = max(high, np.maximum(row * lower, row * upper).sum() + matrix[2, 3])
        return low, high


def training_dataset(opt, phase):
    if opt.train_manifest:
        return CalibratedDataset(opt, opt.train_manifest, phase)
    from .TrainDataset import TrainDataset
    return TrainDataset(opt, phase)
