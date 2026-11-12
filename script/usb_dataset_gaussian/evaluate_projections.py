#!/usr/bin/env python3
# Local contribution draft; author and license pending confirmation (see LICENSE.txt).
"""独立读取已保存的浮点投影，统一计算 PSNR/SSIM；不运行重建。"""
import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity


def read_stack(folder):
    """图片用于查看，指标只读取原始浮点数组及对应的视角信息。"""
    images = np.load(folder / 'projections.npy', mmap_mode='r')
    indices = np.load(folder / 'source_indices.npy')
    angles = np.load(folder / 'angles.npy')
    if images.ndim != 3 or not np.issubdtype(images.dtype, np.floating):
        raise ValueError(f'{folder}: 需要 [angle, vertical, horizontal] 浮点数组')
    if len(images) == 0 or indices.shape != (len(images),) or angles.shape != indices.shape:
        raise ValueError(f'{folder}: 数组与视角数量不一致')
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f'{folder}: 视角索引重复')
    if min(images.shape[1:]) < 7 or not np.isfinite(images).all() or not np.isfinite(angles).all():
        raise ValueError(f'{folder}: 数据无效或图片小于 SSIM 的 7×7 窗口')
    return images, indices, angles


def compare(reference, prediction, data_range):
    """逐投影计算，再取算术平均；两侧不截断、不单独归一化。"""
    ref, indices, angles = reference
    pred, pred_indices, pred_angles = prediction
    if ref.shape != pred.shape or not np.array_equal(indices, pred_indices):
        raise ValueError('实测与渲染投影的尺寸或视角顺序不一致')
    if not np.allclose(angles, pred_angles, rtol=0, atol=1e-7):
        raise ValueError('实测与渲染投影的角度不一致')
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError('data_range 必须是有限正数')
    scores = []
    for actual, rendered, index in zip(ref, pred, indices):
        actual, rendered = actual.astype(np.float64), rendered.astype(np.float64)
        mse = float(np.mean((actual - rendered) ** 2))
        psnr = float(10 * np.log10(data_range ** 2 / mse)) if mse else float('inf')
        ssim = float(structural_similarity(actual, rendered, data_range=data_range,
                                           win_size=7, gaussian_weights=False, use_sample_covariance=True))
        scores.append({'source_index': int(index), 'mse': mse, 'psnr_db': psnr, 'ssim': ssim})
    mean_mse = float(np.mean([s['mse'] for s in scores]))
    return {'mse_mean': mean_mse,
            'psnr_pooled_db': float(10 * np.log10(data_range ** 2 / mean_mse)) if mean_mse else float('inf'),
            'psnr_mean_db': float(np.mean([s['psnr_db'] for s in scores])),
            'ssim_mean': float(np.mean([s['ssim'] for s in scores])), 'per_view': scores}


def json_safe(value):
    # 完全一致的投影 PSNR 为正无穷，用字符串保存为合法 JSON。
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and np.isinf(value):
        return 'Infinity' if value > 0 else '-Infinity'
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='一个算法结果下的 projections 目录')
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--data-range', type=float, required=True, help='所有算法统一的动态范围')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    test = read_stack(args.data_root / 'test_measured')
    train = read_stack(args.data_root / 'train_measured')
    trains = {len(train[0]): train}
    for n, train in trains.items():
        if np.intersect1d(train[1], test[1]).size:
            raise ValueError(f'{n} 视角组的训练与测试索引重叠')
        delta = (train[2][:, None] - test[2][None, :] + np.pi) % (2 * np.pi) - np.pi
        if np.any(np.abs(delta) < 1e-6):
            raise ValueError(f'{n} 视角组的训练与测试存在重复角度')
    references = [test[0]] + [v[0] for v in trains.values()]
    low = min(float(v.min()) for v in references)
    high = max(float(v.max()) for v in references)
    dynamic_range = args.data_range if args.data_range is not None else high - low
    results = {}
    for n, train in trains.items():
        results[str(n)] = {}
        folder = args.data_root
        methods = args.methods
        for method in methods:
            results[str(n)][method] = {}
            for split, reference in [('train', train), ('test', test)]:
                prediction = read_stack(folder / f'{split}_{method}')
                results[str(n)][method][split] = compare(reference, prediction, dynamic_range)
    record = {'data_root': str(args.data_root.resolve()), 'data_range': dynamic_range,
              'data_range_source': 'explicit' if args.data_range is not None else 'all reference max - min',
              'reference_range': [low, high], 'aggregation': 'arithmetic mean over views',
              'ssim': {'window': 7, 'gaussian_weights': False, 'sample_covariance': True},
              'note': '实测投影为参考；不裁剪、不逐图归一化；不是三维真值指标。', 'results': results}
    output = args.output or args.data_root.parent / 'projection_metrics.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(record), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'已保存指标：{output}')


if __name__ == '__main__':
    main()
