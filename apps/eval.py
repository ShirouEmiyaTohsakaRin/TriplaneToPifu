"""Reconstruct with the hybrid Occupancy model and optional original RGB decoder."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from lib.options import BaseOptions
from lib.checkpoint import device_for, read_checkpoint, restore_options, set_projection
from lib.data.CalibratedDataset import CalibratedDataset
from lib.model.triplane_pifu.HybridPIFuNet import HybridPIFuNet
from lib.model.ResBlkPIFuNet import ResBlkPIFuNet
from lib.train_util import gen_mesh, gen_mesh_color


class Evaluator:
    def __init__(self, opt, projection_mode=None):
        self.opt = opt
        if not opt.load_netG_checkpoint_path:
            raise ValueError('Provide --load_netG_checkpoint_path for the trained hybrid model.')
        if opt.init_pifu_checkpoint:
            raise ValueError('--init_pifu_checkpoint is for shape training, not reconstruction.')
        checkpoint = read_checkpoint(opt.load_netG_checkpoint_path)
        restore_options(opt, checkpoint)
        color_checkpoint = read_checkpoint(opt.load_netC_checkpoint_path)
        restore_options(opt, color_checkpoint, kind='color')
        self.dataset = None
        if opt.eval_manifest:
            self.dataset = CalibratedDataset(opt, opt.eval_manifest)
            set_projection(opt, self.dataset.projection_mode)
        elif opt.num_views != 1 or opt.projection_mode == 'perspective':
            raise ValueError('Multi-view/perspective reconstruction requires --eval_manifest with actual camera calibrations.')
        else:
            set_projection(opt, 'orthogonal')
        if projection_mode is not None and projection_mode != opt.projection_mode:
            raise ValueError('Projection argument conflicts with checkpoint/data.')
        self.cuda = device_for(opt)
        self.netG = HybridPIFuNet(opt, opt.projection_mode).to(self.cuda)
        self.netG.load_state_dict(checkpoint['state_dict'], strict=True)
        self.netG.eval()
        self.netC = None
        if color_checkpoint is not None:
            self.netC = ResBlkPIFuNet(opt, opt.projection_mode).to(self.cuda)
            self.netC.load_state_dict(color_checkpoint['state_dict'], strict=True)
            self.netC.eval()
        self.load_size = opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize((self.load_size, self.load_size)), transforms.ToTensor(),
            transforms.Normalize((.5,) * 3, (.5,) * 3),
        ])
        self.output_dir = Path(opt.results_path) / opt.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'opt.txt').write_text(json.dumps(vars(opt), indent=2))
        print('Using hybrid_pifu:', opt.projection_mode, 'views:', opt.num_views,
              'learned color:', self.netC is not None)

    def load_image(self, image_path, mask_path):
        """Original PIFu single-image normalized orthographic convention."""
        with Image.open(image_path) as image:
            image = self.to_tensor(image.convert('RGB'))
        with Image.open(mask_path) as mask:
            mask = transforms.ToTensor()(mask.convert('L').resize(
                (self.load_size, self.load_size), Image.Resampling.NEAREST))
        calib = torch.eye(4)
        calib[1, 1] = -1
        return dict(name=Path(image_path).stem, img=(image * mask).unsqueeze(0),
                    calib=calib.unsqueeze(0), mask=mask.unsqueeze(0),
                    b_min=np.array([-1., -1., -1.]), b_max=np.array([1., 1., 1.]))

    def eval(self, data, use_octree=False):
        path = str(self.output_dir / ('result_' + data['name'] + '.obj'))
        with torch.no_grad():
            if self.netC is None:
                gen_mesh(self.opt, self.netG, self.cuda, data, path, use_octree=use_octree, strict=True)
            else:
                gen_mesh_color(self.opt, self.netG, self.netC, self.cuda, data, path,
                               use_octree=use_octree, strict=True)
        print('Saved:', path)
        return path


def main(opt):
    evaluator = Evaluator(opt)
    if evaluator.dataset is not None:
        for data in evaluator.dataset:
            evaluator.eval(data, use_octree=not opt.no_octree)
        return
    if opt.img_path:
        paths = [Path(opt.img_path)]
    elif opt.test_folder_path:
        paths = sorted(p for p in Path(opt.test_folder_path).iterdir()
                       if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and not p.stem.endswith('_mask'))
    else:
        raise ValueError('Provide --test_folder_path, --img_path or --eval_manifest.')
    if not paths:
        raise ValueError('No input images found.')
    for path in paths:
        mask = opt.mask_path if opt.img_path and opt.mask_path else path.with_name(path.stem + '_mask.png')
        evaluator.eval(evaluator.load_image(path, mask), use_octree=not opt.no_octree)


if __name__ == '__main__':
    main(BaseOptions().parse())
