#!/usr/bin/env python3
# Local contribution draft; author and license pending confirmation (see LICENSE.txt).
"""从单算法三维结果导出逐视角实测/渲染投影，不计算指标。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from reconstruct_cil import load_data
from cil.plugins.tigre import ProjectionOperator


def save_images(folder, images, indices, angles, window):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / 'projections.npy', images)
    np.save(folder / 'source_indices.npy', indices)
    np.save(folder / 'angles.npy', angles)
    for image, index in zip(images, indices):
        # NPY 原样保存；PNG 只用于显示，统一窗口并翻转到图像显示方向。
        np.save(folder / f'{int(index):04d}.npy', image)
        pixels = np.clip((image - window[0]) / (window[1] - window[0]), 0, 1)
        Image.fromarray(np.rint(pixels[::-1] * 255).astype(np.uint8)).save(folder / f'{int(index):04d}.png')


def export_projections(reconstruction, split_manifest):
    """保存一组独立算法结果的所有逐视角投影。"""
    record = json.loads((reconstruction / 'manifest.json').read_text())
    split = json.loads(split_manifest.read_text())
    method = record['method']
    source = Path(record['source'])
    indices = np.load(reconstruction / 'view_indices.npy')
    data, ig, rows, train_indices, _ = load_data(
        source, indices, record['binning'], max(record['volume_shape']))
    np.testing.assert_array_equal(rows, np.load(reconstruction / 'selected_rows.npy'))
    test_rows = np.asarray(split['test_rows'], dtype=int)
    if np.intersect1d(rows, test_rows).size:
        raise ValueError('训练与测试视角重叠')
    angles = np.load(source / 'angles.npy')
    original = np.load(source / 'view_indices.npy')
    np.testing.assert_array_equal(original[test_rows], split['test_source_indices'])
    delta = (angles[rows, None] - angles[test_rows][None, :] + np.pi) % (2 * np.pi) - np.pi
    if np.any(np.abs(delta) < 1e-6):
        raise ValueError('训练与测试存在重复角度')
    raw = np.load(source / 'projections.npy', mmap_mode='r')
    b = record['binning']
    _, h, w = raw.shape
    test_ref = np.stack([raw[i].reshape(h // b, b, w // b, b).mean(axis=(1, 3)) for i in test_rows])
    test_geometry = data.geometry.copy()
    test_geometry.set_angles(angles[test_rows], initial_angle=data.geometry.config.angles.initial_angle,
                             angle_unit='radian')
    volume = np.load(reconstruction / f'{method}_volume.npy')
    if volume.shape != ig.shape or not np.isfinite(volume).all():
        raise ValueError('体积尺寸或数值无效')
    image = ig.allocate(0)
    image.fill(volume)
    output = reconstruction / 'projections'
    if output.exists():
        raise FileExistsError(output)
    window = split['png_window']
    for name, ref, geometry, indices in [('train', data.as_array(), data.geometry, train_indices),
                                        ('test', test_ref, test_geometry, original[test_rows])]:
        rendered = ProjectionOperator(ig, geometry).direct(image).as_array()
        if not np.isfinite(rendered).all():
            raise ValueError('渲染结果存在无效数值')
        save_images(output / f'{name}_measured', ref, indices, geometry.angles, window)
        save_images(output / f'{name}_{method}', rendered, indices, geometry.angles, window)
    (output / 'manifest.json').write_text(json.dumps({
        'method': method, 'split_manifest': str(split_manifest.resolve()),
        'png_window': window, 'png_vertical_flip': True,
        'npy_axis_order': 'vertical, horizontal', 'npy_dtype': 'float32',
        'note': '逐张 NPY 与 projections.npy 数值一致；指标使用 NPY，不使用 PNG。'}, indent=2) + '\n')
    print(f'逐张图片已保存：{output}')
