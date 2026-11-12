#!/usr/bin/env python3
# Local contribution draft; author and license pending confirmation (see LICENSE.txt).
"""复用 CIL 读取 USB 扫描：平场/偏移校正 → 负对数 → 裁剪 → 保存。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'dataset/usb/gruppe 4/gruppe 4_2014-03-20_1404_12/tomo-A/gruppe 4_tomo-A.txrm'


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=ROOT / 'dataset/usb_cil')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--view-start', type=int, default=0)
    p.add_argument('--view-stop', type=int, default=None, help='结束视角索引（不包含），默认读取到末尾')
    p.add_argument('--view-step', type=int, default=1)
    p.add_argument('--crop-horizontal', type=int, default=200, help='偏移校正后，左右各裁剪多少像素')
    p.add_argument('--min-intensity', type=float, default=1e-6, help='取负对数前的透射率下限')
    p.add_argument('--initial-angle-deg', type=float, default=0, help='可选朝向偏移；教程用 -11.5 度调整显示朝向')
    a = p.parse_args()
    if a.batch_size < 1 or a.view_step < 1 or a.view_start < 0 or a.crop_horizontal < 0:
        p.error('Batch size and step must be positive; start/crop must be nonnegative')
    if not 0 < a.min_intensity < 1:
        p.error('--min-intensity must be between 0 and 1')
    return a


def geometry_record(g):
    """保存 CIL 原始坐标系和单位，供后续高斯重建适配使用。"""
    cfg = g.config
    record = {'geometry_type': str(g.geom_type), 'dimension_labels': list(g.dimension_labels),
              'units': str(cfg.units),
              'panel': {'num_pixels': list(map(int, cfg.panel.num_pixels)),
                        'pixel_size': list(map(float, cfg.panel.pixel_size)), 'origin': str(cfg.panel.origin)},
              'angles': {'values': g.angles.tolist(), 'unit': str(cfg.angles.angle_unit),
                         'initial_angle': float(cfg.angles.initial_angle)}, 'system': {}}
    for name in ('source', 'detector', 'rotation_axis', 'ray'):
        obj = getattr(cfg.system, name, None)
        if obj is not None:
            record['system'][name] = {key: getattr(obj, key).tolist()
                                     for key in ('position', 'direction', 'direction_x', 'direction_y')
                                     if hasattr(obj, key)}
    return record


def main():
    args = arguments()
    import numpy as np
    from cil.version import version as cil_version
    from cil.io import ZEISSDataReader
    from cil.processors import TransmissionAbsorptionConverter, Slicer

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f'Output exists; choose a new --output: {output}')
    stage = output.with_name(output.name + '.incomplete')
    if stage.exists():
        raise FileExistsError(f'Previous incomplete output exists: {stage}')
    # 先读取元数据，检查参考图和角度范围。
    reader = ZEISSDataReader(file_name=str(source))
    metadata = reader.get_metadata()
    count = int(metadata['number_of_images'])
    stop = count if args.view_stop is None else args.view_stop
    if not args.view_start < stop <= count:
        raise ValueError(f'Expected 0 <= view-start < view-stop <= {count}')
    if args.crop_horizontal * 2 >= metadata['image_width']:
        raise ValueError('Crop removes the entire detector width')
    indices = np.arange(args.view_start, stop, args.view_step, dtype=np.int64)
    if not np.isfinite(args.initial_angle_deg):
        raise ValueError('Initial angle must be finite')
    reference = np.asarray(metadata['reference'])
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError('Invalid flat reference; refusing division by zero or invalid values')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    # CIL 已做参考图归一化，因此 white_level 固定为 1，不重复做平场。
    converter = TransmissionAbsorptionConverter(min_intensity=args.min_intensity, white_level=1.0, accelerated=False)
    projections = None
    full_geometry = None
    clipped_pixels = 0
    for begin in range(0, len(indices), args.batch_size):
        selected = indices[begin:begin + args.batch_size]
        # 只分批选择角度，保留完整图像做偏移校正，随后再裁剪。
        data = ZEISSDataReader(file_name=str(source),
                               roi={'angle': (int(selected[0]), int(selected[-1]) + 1, args.view_step)}).read()
        transmission = data.as_array()
        if not np.isfinite(transmission).all():
            raise ValueError(f'Non-finite transmission at batch {begin}')
        clipped_pixels += int(np.count_nonzero(transmission < args.min_intensity))
        # 只取一次负对数。保留负衰减值，不去噪、不强制截断到零。
        data = converter(data)
        if args.crop_horizontal:
            data = Slicer(roi={'horizontal': (args.crop_horizontal, -args.crop_horizontal)})(data)
        # CIL 会省略单角度的维度，这里统一还原为 [角度, 高, 宽]。
        array = np.asarray(data.as_array(), dtype=np.float32)
        array = array.reshape(len(selected), *array.shape[-2:])
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite absorption at batch {begin}')
        if projections is None:
            projections = np.lib.format.open_memmap(stage / 'projections.npy', mode='w+', dtype=np.float32,
                                                   shape=(len(indices), *array.shape[1:]))
            full_geometry = data.geometry.copy()
            full_geometry.set_angles(np.asarray(metadata['thetas'])[indices],
                                     initial_angle=np.deg2rad(args.initial_angle_deg), angle_unit='radian')
        expected_angles = np.asarray(metadata['thetas'])[selected]
        np.testing.assert_allclose(data.geometry.angles, expected_angles)
        projections[begin:begin + len(selected)] = array
        print(f'Processed {begin + len(selected)}/{len(indices)} views', flush=True)
    # 全部批次成功后再发布目录，避免半成品被当成可用数据。
    projections.flush()
    shape = list(projections.shape)
    del projections
    np.save(stage / 'angles.npy', np.asarray(full_geometry.angles))
    np.save(stage / 'view_indices.npy', indices)
    (stage / 'geometry.json').write_text(json.dumps(geometry_record(full_geometry), indent=2) + '\n')
    manifest = {'source': str(source), 'source_size': source.stat().st_size,
                'cil_version': cil_version,
                'shape': shape, 'dtype': 'float32', 'dimension_labels': ['angle', 'vertical', 'horizontal'],
                'source_views': count, 'processed_views': len(indices),
                'clipped_pixels_before_crop': clipped_pixels,
                'processing': {'reader': 'CIL ZEISSDataReader: reference division and stored integer shifts',
                               'converter': 'CIL TransmissionAbsorptionConverter', 'white_level': 1.0,
                               'min_intensity': args.min_intensity, 'crop_each_side': args.crop_horizontal,
                               'initial_angle_deg': args.initial_angle_deg, 'denoising': False,
                               'negative_absorption_preserved': True},
                'format': 'CIL acquisition export; not yet a FaCT-GS/R2-Gaussian training dataset'}
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stage.rename(output)
    print(f'COMPLETE: {output} {shape}', flush=True)


if __name__ == '__main__':
    main()
