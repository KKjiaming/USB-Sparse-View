"""Notebook 的几何辅助函数；FDK/SIRT 算法直接写在 Notebook 中。"""
import json
from pathlib import Path
import numpy as np
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry


def load_data(source, indices, binning=4, resolution=128):
    """读取已预处理的衰减积分；只降采样，不重复平场或取对数。"""
    source = Path(source).resolve()
    g = json.loads((source / 'geometry.json').read_text())
    projections = np.load(source / 'projections.npy', mmap_mode='r')
    angles = np.load(source / 'angles.npy')
    original_indices = np.load(source / 'view_indices.npy')
    n, height, width = projections.shape
    if g['geometry_type'] != 'CONE' or g['panel']['num_pixels'] != [width, height]:
        raise ValueError('需要尺寸一致的 CIL 锥束数据')
    if g['angles']['unit'] != 'radian' or g['panel']['origin'] != 'bottom-left':
        raise ValueError('需要弧度角度和 bottom-left 探测器原点')
    if len(angles) != n or len(original_indices) != n or not np.isfinite(angles).all():
        raise ValueError('投影、角度与索引不一致')
    np.testing.assert_allclose(angles, g['angles']['values'])
    b = binning
    if height % b or width % b:
        raise ValueError('投影尺寸必须能被 binning 整除')
    # 直接复用实验划分中的原始索引，不另写一套视角抽样规则。
    by_index = {int(index): row for row, index in enumerate(original_indices)}
    selected = np.array([by_index[int(index)] for index in indices], dtype=int)
    if len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError('训练索引为空或重复')
    images = []
    for i in selected:
        image = np.asarray(projections[i], dtype=np.float32)
        images.append(image.reshape(height // b, b, width // b, b).mean(axis=(1, 3)))
    images = np.stack(images)
    if not np.isfinite(images).all():
        raise ValueError('输入投影存在无效值')
    system = g['system']
    ag = AcquisitionGeometry.create_Cone3D(
        source_position=system['source']['position'], detector_position=system['detector']['position'],
        detector_direction_x=system['detector']['direction_x'], detector_direction_y=system['detector']['direction_y'],
        rotation_axis_position=system['rotation_axis']['position'], rotation_axis_direction=system['rotation_axis']['direction'])
    ag.set_panel([width // b, height // b], pixel_size=np.asarray(g['panel']['pixel_size']) * b,
                 origin='bottom-left')
    ag.set_angles(angles[selected], initial_angle=g['angles']['initial_angle'], angle_unit='radian')
    pixel_x, pixel_y = g['panel']['pixel_size']
    sizes = np.array([width * pixel_x, width * pixel_x, height * pixel_y]) / ag.magnification
    counts = np.maximum(2, np.rint(sizes / max(sizes) * resolution).astype(int))

    # 与 FaCT-GS 使用相同的最长边 2 的坐标尺度；衰减积分保持不变。
    scale = 2 / max(sizes)
    for obj in (ag.config.system.source, ag.config.system.detector, ag.config.system.rotation_axis):
        obj.position = obj.position * scale
    ag.config.panel.pixel_size *= scale
    ag.config.units = 'normalized distance'
    spacing = sizes * scale / counts
    ig = ImageGeometry(voxel_num_x=int(counts[0]), voxel_num_y=int(counts[1]), voxel_num_z=int(counts[2]),
                       voxel_size_x=spacing[0], voxel_size_y=spacing[1], voxel_size_z=spacing[2])
    data = AcquisitionData(images, geometry=ag)
    return data, ig, selected, original_indices[selected], scale


def save_slices(volumes, output):
    """只对图片使用统一灰度窗；保存的浮点体积不截断、不归一化。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    samples = np.concatenate([v.ravel()[::16] for v in volumes.values()])
    low, high = np.percentile(samples, [1, 99.5])
    fig, axes = plt.subplots(len(volumes), 3, figsize=(12, 4 * len(volumes)), squeeze=False)
    for row, (method, volume) in enumerate(volumes.items()):
        for axis, label in enumerate(['Z', 'Y', 'X']):
            image = np.take(volume, volume.shape[axis] // 2, axis=axis)
            axes[row, axis].imshow(image, cmap='gray', vmin=low, vmax=high, origin='lower')
            axes[row, axis].set_title(f'{method.upper()} | {label} centre')
            axes[row, axis].axis('off')
    fig.tight_layout()
    fig.savefig(output / 'slices.png', dpi=160)
    plt.close(fig)
    return [float(low), float(high)]
