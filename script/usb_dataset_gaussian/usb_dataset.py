"""USB 数据集接口：读取已转换的投影，适配 FaCT-GS SceneInfo。"""
from pathlib import Path
import json
import numpy as np


class USBDataset:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.metadata = json.loads((self.path / 'meta_data.json').read_text())
        self.conversion = json.loads((self.path / 'conversion.json').read_text())
        meta = self.metadata
        if meta.get('has_volume_ground_truth') is not False or meta.get('vol') is not None:
            raise ValueError('USB 数据集没有三维真值，请使用 convert.py 生成输入')
        if not np.isclose(max(meta['scanner']['sVoxel']), 2):
            raise ValueError('扫描几何应归一化到最长边 2')
        self.indices = {}
        for split in ['train', 'test']:
            frames = meta[f'proj_{split}']
            indices = [f['source_index'] for f in frames]
            if not indices or len(set(indices)) != len(indices):
                raise ValueError(f'{split} 视角为空或索引重复')
            if indices != self.conversion[f'{split}_indices']:
                raise ValueError(f'{split} 元数据与转换记录中的索引不一致')
            for frame in frames:
                image = np.load(self.path / frame['file_path'], mmap_mode='r')
                if image.shape != tuple(meta['scanner']['nDetector']) or not np.isfinite(image).all():
                    raise ValueError(f'无效投影：{frame["file_path"]}')
            self.indices[split] = indices
        if set(self.indices['train']) & set(self.indices['test']):
            raise ValueError('训练／测试视角重叠')

    def scene_info(self, evaluate=True):
        # 延迟导入：由训练入口先配置 FaCT-GS 路径和 CUDA 库。
        from fact_gs.r2_gaussian.dataset.dataset_readers import SceneInfo, readCTameras
        cameras = readCTameras(self.metadata, str(self.path), evaluate, scene_scale=1.0)
        return SceneInfo(cameras['train'], cameras['test'], None, self.metadata['scanner'], 1.0)


def read_usb_scene(path, evaluate):
    """符合 FaCT-GS 数据集读取回调签名，不重复缩放投影。"""
    return USBDataset(path).scene_info(evaluate)
