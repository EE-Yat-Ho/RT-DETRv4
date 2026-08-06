"""
Repeat Factor Sampling (RFS) for class-imbalanced detection datasets.

Reference:
    Gupta et al., "LVIS: A Dataset for Large Vocabulary Instance Segmentation",
    CVPR 2019, Sec. 4.1 ("Repeat factor sampling").

Opt-in: nothing here runs unless `train_dataloader.sampler` is present in the
config. Removing that config block restores the previous behaviour exactly.

Self-contained by design -- this module imports nothing from the rest of
`engine`, so it can be deleted together with its config block and the small
call sites in `dataloader.py` / `dist_utils.py` / `yaml_config.py`.
"""

import math
from collections import Counter

import torch
import torch.utils.data as data


__all__ = ['RepeatFactorSampler', 'build_sampler']


def _dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0


class RepeatFactorSampler(data.Sampler):
    """Oversamples images containing rare categories.

        f(c)  = (# images containing c) / (# images)
        r(c)  = max(1, (t / f(c)) ** power)
        r(I)  = max{ r(c) | c in I }

    Every quantity is derived from the annotation file at construction time, so
    adding categories or changing their counts needs no config change.

    Args:
        dataset: a `CocoDetection` (anything exposing `.coco` and `.ids`).
        repeat_thresh: `t` above. When <= 0 it is set to max_c f(c), i.e. the
            most frequent category gets r(c) == 1 and everything else is pulled
            up towards it. This is the self-tuning default -- a fixed value such
            as the LVIS 0.001 is meaningless on datasets whose rarest category
            is still well above that frequency.
        power: exponent on t/f(c). 0.5 (sqrt) is the LVIS default and only
            partially closes the gap; 1.0 aims at full balance but repeats the
            same rare images hard enough to invite overfitting.
        max_repeat: optional clamp on r(c). <= 0 disables it.
        shuffle: shuffle the repeated index list each epoch.
        seed: base seed; the per-epoch seed is `seed + epoch` so that all ranks
            build the identical index list before slicing it.
    """

    def __init__(self,
                 dataset,
                 repeat_thresh: float = -1.0,
                 power: float = 0.5,
                 max_repeat: float = -1.0,
                 shuffle: bool = True,
                 seed: int = 0,
                 verbose: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.num_replicas, self.rank = _dist_info()

        img_cats = self._image_categories(dataset)
        num_images = len(img_cats)
        if num_images == 0:
            raise ValueError('RepeatFactorSampler: dataset is empty.')

        cat_img_count = Counter()
        for cats in img_cats:
            cat_img_count.update(cats)
        if not cat_img_count:
            raise ValueError('RepeatFactorSampler: no annotations found; '
                             'cannot compute repeat factors.')

        freq = {c: n / num_images for c, n in cat_img_count.items()}
        thresh = repeat_thresh if repeat_thresh > 0 else max(freq.values())

        cat_repeat = {}
        for c, f in freq.items():
            r = max(1.0, (thresh / f) ** power)
            if max_repeat > 0:
                r = min(r, max_repeat)
            cat_repeat[c] = r

        # Images with no annotations keep r == 1: they are useful negatives and
        # must not be dropped.
        repeat = torch.ones(num_images, dtype=torch.float64)
        for i, cats in enumerate(img_cats):
            if cats:
                repeat[i] = max(cat_repeat[c] for c in cats)
        self._repeat = repeat

        total = float(repeat.sum())
        self.num_samples = math.ceil(total / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        if verbose and self.rank == 0:
            self._log(dataset, cat_img_count, freq, cat_repeat, thresh,
                      num_images, total, repeat_thresh, power)

    @staticmethod
    def _image_categories(dataset):
        coco = getattr(dataset, 'coco', None)
        ids = getattr(dataset, 'ids', None)
        if coco is None or ids is None:
            raise TypeError(
                'RepeatFactorSampler needs a COCO-style dataset exposing '
                f'`.coco` and `.ids`, got {type(dataset).__name__}.')

        img_cats = []
        for img_id in ids:
            anns = coco.imgToAnns.get(img_id, []) or []
            # Match the iscrowd filtering done in ConvertCocoPolysToMask so the
            # frequencies reflect what the model is actually trained on.
            img_cats.append({a['category_id'] for a in anns
                             if not a.get('iscrowd', 0)})
        return img_cats

    def _log(self, dataset, cat_img_count, freq, cat_repeat, thresh,
             num_images, total, repeat_thresh, power):
        try:
            names = {c['id']: c['name'] for c in dataset.coco.dataset['categories']}
        except Exception:
            names = {}

        mode = 'auto = max(f)' if repeat_thresh <= 0 else 'fixed'
        print(f'  ## RepeatFactorSampler: t={thresh:.4f} ({mode}), power={power} ##')
        print(f'  {"category":<20}{"imgs":>10}{"f(c)":>10}{"r(c)":>8}')
        for c in sorted(freq, key=lambda k: -freq[k]):
            print(f'  {names.get(c, c):<20}{cat_img_count[c]:>10}'
                  f'{freq[c]:>10.4f}{cat_repeat[c]:>8.2f}')
        print(f'  epoch size: {num_images} -> {int(total)} images '
              f'(x{total / num_images:.3f})')

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _epoch_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Stochastic rounding: an r of 1.93 yields 1 copy 7% of the time and 2
        # copies 93% of the time, so the expectation over epochs is exactly r.
        floor = torch.floor(self._repeat)
        frac = self._repeat - floor
        counts = (floor + (torch.rand(len(self._repeat), generator=g) < frac)).long()

        indices = torch.repeat_interleave(torch.arange(len(counts)), counts)
        if self.shuffle:
            indices = indices[torch.randperm(len(indices), generator=g)]

        # `total_size` is fixed so that __len__ stays constant across epochs and
        # every rank gets the same number of batches.
        while len(indices) < self.total_size:
            indices = torch.cat([indices, indices])
        return indices[:self.total_size]

    def __iter__(self):
        indices = self._epoch_indices()[self.rank::self.num_replicas]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


_SAMPLERS = {
    'RepeatFactorSampler': RepeatFactorSampler,
}


def build_sampler(cfg: dict, dataset, shuffle: bool = True):
    """Instantiate a sampler from a config dict, injecting the built dataset."""
    cfg = dict(cfg)
    name = cfg.pop('type', None)
    if name not in _SAMPLERS:
        raise ValueError(f'Unknown sampler type {name!r}; '
                         f'expected one of {sorted(_SAMPLERS)}.')
    cfg.setdefault('shuffle', shuffle)
    return _SAMPLERS[name](dataset, **cfg)
