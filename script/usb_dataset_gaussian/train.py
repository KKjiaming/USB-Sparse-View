#!/usr/bin/env python3
# Local contribution draft; author and license pending confirmation (see LICENSE.txt).
"""复用 FaCT-GS 训练 USB；没有体积真值，仅评估二维投影。"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'third_party/fact-gs'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/usb_fact_gs_50views')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--gaussians', type=int, default=10000)
    args = parser.parse_args()
    if args.steps < 1 or args.gaussians < 1:
        parser.error('steps 和 gaussians 必须为正数')
    data_path, output = args.data.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'输出已存在：{output}')
    os.environ['GLM_HOME'] = str(REPO / 'fact_gs/submodules/glm')
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    # 与现有训练入口一致，先导入 TIGRE，再导入 Torch。
    import tigre
    import numpy as np
    import torch
    import train_recon as trainer
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from usb_dataset import USBDataset, read_usb_scene
    from fact_gs.r2_gaussian.utils.general_utils import safe_state

    dataset = USBDataset(data_path)
    meta = dataset.metadata
    conversion = dataset.conversion
    split_path = conversion.get('split_manifest')
    if not split_path:
        raise ValueError('实验训练需要 convert.py --split-manifest 固定视角划分')
    split_record = json.loads(Path(split_path).read_text())
    from export_cil_projections import save_images
    from reconstruct_cil import save_slices
    history = []

    @torch.no_grad()
    def report(step, losses, training_time, eval_time, max_steps, eval_args,
               scene, render, voxelize, init_mode, force_eval=False):
        # 只记录优化 loss；PSNR/SSIM 留给独立指标脚本。
        if step % 100 == 0 or step == max_steps:
            history.append({'step': step, 'training_seconds': training_time,
                            **{k: float(v) for k, v in losses.items() if k.startswith('loss_')}})
            (output / 'training_loss.json').write_text(json.dumps(history, indent=2) + '\n')
        if step != max_steps:
            return None
        for split, cameras in [('train', scene.getTrainCameras()), ('test', scene.getTestCameras())]:
            # 明确按元数据索引配对，不依赖训练相机的遍历顺序。
            by_index = {int(c.image_name): c for c in cameras}
            frames = meta[f'proj_{split}']
            observed, predicted = [], []
            for frame in frames:
                camera = by_index[frame['source_index']]
                prediction = render(camera, scene.gaussians)['render'].squeeze().cpu().numpy()
                observation = camera.original_image.squeeze().cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError('渲染结果包含非有限值')
                # 转回 CIL 的 vertical 方向，使保存数组与传统算法逐像素对应。
                predicted.append(np.ascontiguousarray(prediction[::-1], dtype=np.float32))
                observed.append(np.ascontiguousarray(observation[::-1], dtype=np.float32))
            indices = np.array([f['source_index'] for f in frames])
            angles = np.array([f['cil_angle'] for f in frames])
            for name, values in [('measured', observed), ('factgs', predicted)]:
                save_images(output / f'projections/{split}_{name}', np.stack(values),
                            indices, angles, split_record['png_window'])
        volume = voxelize(scene.gaussians)['vol'].cpu().numpy().astype(np.float32)
        # 当前 FaCT-GS voxelizer 实际返回 XYZ，转为 CIL 的 ZYX。
        volume = np.ascontiguousarray(volume.transpose(2, 1, 0))
        expected = tuple(meta['scanner']['nVoxel'][::-1])
        if volume.shape != expected or not np.isfinite(volume).all():
            raise ValueError(f'无效的体积：{volume.shape}, expected {expected}')
        np.save(output / 'factgs_volume.npy', volume)
        save_slices({'factgs': volume}, output)
        record = {'method': 'factgs', 'steps': max_steps, 'initial_gaussians': args.gaussians,
                  'final_gaussians': int(scene.gaussians.get_xyz.shape[0]),
                  'training_seconds': training_time, 'data': str(data_path),
                  'train_indices': conversion['train_indices'], 'test_indices': conversion['test_indices'],
                  'volume_axis_order': 'ZYX', 'volume_shape': list(volume.shape),
                  'voxel_size_xyz': meta['scanner']['dVoxel'], 'geometry_scale': conversion['geometry_scale'],
                  'projection_axis_order': ['angle', 'vertical', 'horizontal'],
                  'projection_orientation': 'CIL bottom-left; reversed from FaCT-GS camera rows',
                  'png_window': split_record['png_window'], 'seed': 0,
                  'metrics_status': 'pending; evaluate saved float32 projections independently'}
        (output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r['step'] for r in history], [r['loss_total'] for r in history])
        ax.set(xlabel='Training step', ylabel='Sampled total loss')
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(output / 'training_loss.png', dpi=160)
        plt.close(fig)
        return None

    # 显式传入 USB 读取和结果保存函数，不修改全局回调表。
    with initialize_config_dir(config_dir=str(REPO / 'config'), version_base=None):
        config = compose(config_name='default_recon', overrides=[
            f'model.data_source_path={data_path}', f'model.model_path={output}',
            f'model.num_gaussians={args.gaussians}', 'model.init_mode=gradient',
            'model.density_thresh=0.01', f'optim.steps={args.steps}', 'eval=eval_silent'])
    output.mkdir(parents=True)
    OmegaConf.save(config, output / 'config.yaml')
    safe_state(False)
    started = time.perf_counter()
    trainer.optimize(config, scene_reader=read_usb_scene, report_callback=report)
    manifest_path = output / 'manifest.json'
    record = json.loads(manifest_path.read_text())
    record['wall_seconds'] = time.perf_counter() - started
    manifest_path.write_text(json.dumps(record, indent=2) + '\n')
    (output / 'COMPLETE').write_text('训练完成；体积与逐张浮点投影已保存，评价指标另算。\n')


if __name__ == '__main__':
    main()
