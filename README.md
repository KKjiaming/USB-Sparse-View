# USB sparse-view CT: CIL and FaCT-GS

This folder is the complete **source-code case**: the notebook can run FDK, SIRT and FaCT-GS directly. It does not import or run code from the parent project's `code/fact-gs` directory. No separate FaCT-GS checkout is required.

## Structure

- `USB_Sparse_View.ipynb`: step-by-step execution, with FDK/SIRT directly in cells.
- `script/usb_dataset_gaussian/`: small USB-specific helpers and the training entry point.
- `third_party/fact-gs/`: bundled FaCT-GS source and all seven submodules as ordinary files, including CUDA extension sources and GLM headers.
- `third_party/SOURCES.json`: upstream repository URLs, commit IDs and original file hashes.
- `third_party/usb_hooks.patch`: the small upstream changes that accept the USB dataset reader and result writer as explicit arguments. The optimizer and loss are unchanged.
- `assets/` and `EXPERIMENT_RECORD.md`: fixed splits and historical results; no automatic image display.

The USB helper scripts are separated by task:

| File | Responsibility |
| --- | --- |
| `prepare.py` | Read the raw TXRM with CIL and save preprocessed projections |
| `reconstruct_cil.py` | Build geometry and load fixed view indices; no duplicate FDK/SIRT runner |
| `export_cil_projections.py` | Forward-project saved volumes and save every NPY/PNG |
| `convert.py` | Convert the fixed split to Gaussian camera inputs |
| `usb_dataset.py` | Read USB metadata without fabricating volume ground truth |
| `train.py` | Invoke the bundled optimizer and save volume/projection outputs |
| `evaluate_projections.py` | Independently calculate metrics on one method's saved arrays |

## Environment

Linux, an NVIDIA GPU, a compatible driver, and the CUDA toolkit including `nvcc` are needed. The original experiment used CUDA 12.1. From this directory:

```bash
conda env create -f environment.yml
conda activate usb-showcase
bash install_extensions.sh
python -m ipykernel install --user --name usb-showcase --display-name 'Python (USB showcase)'
jupyter lab USB_Sparse_View.ipynb
```

The installation script builds extensions from the bundled source. It does not clone another repository. `environment.yml` records the Python/PyTorch/CIL versions used in the experiment; it has not been validated with a fresh Conda solve. Existing `fact-gs` environments with these dependencies and extensions can also run the notebook. Register/select the corresponding kernel and use it for all cells.

## Data and execution

The source-code bundle does not include the USB scan or large generated arrays. Place the original scan under `dataset/gruppe 4_tomo-A.txrm`, or supply its path. Prepare data once:

```bash
python script/usb_dataset_gaussian/prepare.py \
  --source 'dataset/gruppe 4_tomo-A.txrm' --output dataset/usb_cil
```

The notebook defaults to this case's `dataset/usb_cil`. To reuse an existing prepared dataset, set `SOURCE` in the configuration cell to that dataset directory, or launch Jupyter with `USB_CIL_DATA` set. `USB_TXRM` can similarly specify the raw scan. These are input-data locations, not external source-code dependencies.

1. Select the reconstruction kernel and check `SOURCE` and `GPU`.
2. Start with `VIEWS = 25`. Enable only `RUN_FDK`, then execute setup, geometry, reconstruction and projection export.
3. Execute SIRT separately with `RUN_SIRT`; execute FaCT-GS separately with `RUN_FACTGS`. Gaussian conversion and training are separate cells.
4. Enable `RUN_METRICS`, selecting the completed methods in `METHODS_TO_EVALUATE`. The final table shows only the current run.
5. Repeat for 50 and 75 views. Restart the kernel before changing GPU.

New outputs go to `runs_step_by_step/<views>views/<method>/`. Existing output directories are not overwritten. Use a fresh `RUN_ROOT` for another trial. Every measured/rendered projection is saved as NPY and PNG; metrics read NPY only. The only optional inline image preview is raw preprocessing; loss plots and volumes are saved to disk.

## Redistribution and submission

Submit this **whole folder**, including `third_party/`, `environment.yml` and `install_extensions.sh`. The vendored source includes upstream examples/assets and is about 80 MB; no build products, model outputs, or Git history are bundled. The `.gitignore` does not exclude vendored dataset-reader code.

All upstream license files and copyright notices remain in place. The dependencies have multiple licenses; do not apply FaCT-GS's top-level MIT license to its R²-Gaussian code or all submodules. See `third_party/fact-gs/LICENSE.md` and the individual license files. Author attribution and the license of the new case-specific code still need confirmation, as do the USB dataset's public source/access and reuse terms. `LICENSE.txt` remains a draft marker, not a license grant. This is an independent draft case; no contribution PR has been submitted to CIL-User-Showcase.

## Validation after bundling

The case was copied to an unrelated temporary directory. Using only the copied case source and the existing installed dependency environment:

- Inputs for 25/50/75 views matched the original experiment arrays exactly.
- FaCT-GS conversion, a 20-step/1000-Gaussian smoke training, volume and all projection exports, and independent metrics completed.
- The copied notebook ran 25-view FDK, projection export and metrics through the `fact-gs` Jupyter kernel.
- All bundled upstream files were checked against recorded hashes; only the two documented hook changes differ.

The earlier smoke run checks integration, not convergence. On 2026-11-12, the notebook was additionally executed in three separate `fact-gs` kernels for 25 views: raw preprocessing preview + FDK, SIRT (1000 iterations), and FaCT-GS (20000 steps, 10000 initial Gaussians). All three completed volume/projection export and independent metrics. Every individual NPY/PNG and the common measured inputs were checked. Validation outputs are under `runs_step_by_step/validation_20261112_201233/`; execution records are in the parent project at `output/log/showcase_validation_20261112_201233/`.

Local input links have been added under the ignored `dataset/` directory to the existing USB data. These links are workspace-specific; another machine must still supply its own scan or prepared data. A fresh Conda installation/CUDA rebuild remains untested. Use a fresh `RUN_ROOT` when rerunning to avoid existing-output errors.
