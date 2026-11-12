#!/usr/bin/env python3
# Local contribution draft; author and license pending confirmation (see LICENSE.txt).
"""将 CIL 的 USB 投影导出为 FaCT-GS 使用的扫描几何和逐视角文件。"""
import argparse
import json
from pathlib import Path

import numpy as np
from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.plugins.tigre.Geometry import CIL2TIGREGeometry

ROOT = Path(__file__).resolve().parents[2]


def convert(source, output, train_views, test_views, binning, resolution, split_manifest):
    if output.exists() or output.with_suffix('.incomplete').exists():
        raise FileExistsError(f'输出已存在，请选择新目录：{output}')
    g = json.loads((source / 'geometry.json').read_text())
    projections = np.load(source / 'projections.npy', mmap_mode='r')
    angles = np.load(source / 'angles.npy')
    indices = np.load(source / 'view_indices.npy')
    n, height, width = projections.shape
    if g['geometry_type'] != 'CONE' or g['panel']['num_pixels'] != [width, height]:
        raise ValueError('需要尺寸一致的 CIL 锥束数据')
    if len(angles) != n or len(indices) != n or not np.isfinite(angles).all():
        raise ValueError('投影、角度和索引不一致')
    if height % binning or width % binning:
        raise ValueError('投影尺寸必须能被 binning 整除')
    if g['panel']['origin'] != 'bottom-left' or g['angles']['unit'] != 'radian':
        raise ValueError('目前只支持 bottom-left 探测器原点和弧度角度')
    np.testing.assert_allclose(angles, g['angles']['values'])
    system = g['system']
    ag = AcquisitionGeometry.create_Cone3D(
        source_position=system['source']['position'], detector_position=system['detector']['position'],
        detector_direction_x=system['detector']['direction_x'], detector_direction_y=system['detector']['direction_y'],
        rotation_axis_position=system['rotation_axis']['position'], rotation_axis_direction=system['rotation_axis']['direction'])
    ag.set_panel([width // binning, height // binning],
                 pixel_size=np.asarray(g['panel']['pixel_size']) * binning, origin='bottom-left')
    ag.set_angles(angles, initial_angle=g['angles']['initial_angle'], angle_unit='radian')

    # 使用物体平面的视野定义体积；保持近似各向同性体素。
    pixel_x, pixel_y = g['panel']['pixel_size']
    sizes = np.array([width * pixel_x, width * pixel_x, height * pixel_y]) / ag.magnification
    counts = np.maximum(2, np.rint(sizes / max(sizes) * resolution).astype(int))
    spacing = sizes / counts
    ig = ImageGeometry(voxel_num_x=int(counts[0]), voxel_num_y=int(counts[1]), voxel_num_z=int(counts[2]),
                       voxel_size_x=spacing[0], voxel_size_y=spacing[1], voxel_size_z=spacing[2])
    geo, target_angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)
    # FaCT-GS 的相机仅支持无倾斜、无偏移的圆轨迹；不静默忽略复杂几何。
    for value in [geo.rotDetector, geo.offDetector, geo.offOrigin]:
        if not np.allclose(value, 0, atol=1e-6):
            raise ValueError('当前 FaCT-GS 入口不支持偏移或倾斜的扫描几何')

    # 直接归一化几何到最长边 2，避免官方读取器再次缩放实测衰减积分。
    scale = 2 / max(geo.sVoxel)
    scanner = {'mode': 'cone', 'DSO': float(geo.DSO * scale), 'DSD': float(geo.DSD * scale),
               'nDetector': geo.nDetector.tolist(), 'sDetector': (geo.sDetector * scale).tolist(),
               'dDetector': (geo.dDetector * scale).tolist(), 'nVoxel': geo.nVoxel[::-1].tolist(),
               'sVoxel': (geo.sVoxel[::-1] * scale).tolist(), 'dVoxel': (geo.dVoxel[::-1] * scale).tolist(),
               'offOrigin': [0, 0, 0], 'offDetector': [0, 0], 'accuracy': 0.5, 'filter': None}

    # 展示案例只使用已经固定的实验划分。
    split_record = json.loads(split_manifest.read_text())
    chosen = split_record['experiments'][str(train_views)]
    by_index = {int(index): row for row, index in enumerate(indices)}
    train = np.array([by_index[i] for i in chosen['train_source_indices']])
    test = np.array([by_index[i] for i in split_record['test_source_indices']])
    if len(train) != train_views or len(test) != test_views:
        raise ValueError('固定划分与请求的视角数不一致')
    if np.intersect1d(train, test).size or len(np.unique(train)) != len(train):
        raise ValueError('训练和测试索引重叠或重复')
    delta = (angles[train, None] - angles[test][None, :] + np.pi) % (2 * np.pi) - np.pi
    if np.any(np.abs(delta) < 1e-6):
        raise ValueError('训练与测试存在重复角度')
    stage = output.with_suffix('.incomplete')
    stage.mkdir(parents=True)
    meta = {'scanner': scanner, 'vol': None, 'has_volume_ground_truth': False}
    for split, selected in [('train', train), ('test', test)]:
        (stage / f'proj_{split}').mkdir()
        meta[f'proj_{split}'] = []
        for i in selected:
            # 对已取负对数的投影做块平均：用于降分辨率，不是光子计数合并。
            image = np.asarray(projections[i], dtype=np.float32)
            image = image.reshape(height // binning, binning, width // binning, binning).mean(axis=(1, 3))
            image = np.ascontiguousarray(image[::-1])  # FaCT-GS 内部 FDK 会再翻转回来。
            if not np.isfinite(image).all():
                raise ValueError(f'视角 {i} 存在无效数值')
            filename = f'proj_{split}/{int(indices[i]):04d}.npy'
            np.save(stage / filename, image)
            meta[f'proj_{split}'].append({'file_path': filename, 'angle': float(target_angles[i]),
                                         'source_index': int(indices[i]), 'cil_angle': float(angles[i])})
    (stage / 'meta_data.json').write_text(json.dumps(meta, indent=2) + '\n')
    record = {'source': str(source), 'binning': binning, 'geometry_scale': float(scale),
              'angle_conversion': 'CIL2TIGREGeometry.getTIGREGeometry', 'flip_rows': True,
              'projection_scale': 1.0, 'ground_truth': None,
              'split_manifest': str(split_manifest.resolve()),
              'train_indices': indices[train].tolist(), 'test_indices': indices[test].tolist()}
    (stage / 'conversion.json').write_text(json.dumps(record, indent=2) + '\n')
    stage.rename(output)
    print(f'已保存 {len(train)} 个训练视角、{len(test)} 个测试视角：{output}')
    print('探测器尺寸:', scanner['nDetector'], '体积尺寸:', scanner['nVoxel'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'dataset/usb_cil')
    parser.add_argument('--output', type=Path, default=ROOT / 'dataset/usb_fact_gs')
    parser.add_argument('--train-views', type=int, default=50)
    parser.add_argument('--test-views', type=int, default=20)
    parser.add_argument('--binning', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=128, help='体积最长边的体素数')
    parser.add_argument('--split-manifest', type=Path, required=True, help='复用 CIL 实验的固定视角划分')
    args = parser.parse_args()
    if min(args.train_views, args.test_views, args.binning, args.resolution) < 1:
        parser.error('所有数量参数必须为正数')
    convert(args.source.resolve(), args.output.resolve(), args.train_views, args.test_views, args.binning, args.resolution, args.split_manifest)
