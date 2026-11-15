# 安装与运行

需要 Linux、Conda、NVIDIA GPU 及兼容驱动。编译扩展还需要 CUDA Toolkit（参考版本 11.8，含 `nvcc`）和兼容的 C/C++ 编译器。

在项目目录执行：

```bash
conda env create -f environment.yml
conda activate usb-showcase
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
bash install_extensions.sh
```

Conda 只创建 Python 环境；其余依赖由 pip 安装，见 [requirements.txt](requirements.txt)。安装脚本从 `third_party/fact-gs` 编译 TIGRE 和 FaCT-GS 扩展，请保留该目录。若编译时找不到 CUDA，将 `CUDA_HOME` 指向实际的 Toolkit 安装目录，并将其 `bin` 加入 `PATH`。

打开 [USB_Sparse_View_zh.ipynb](USB_Sparse_View_zh.ipynb)，选择 `usb-showcase` 对应的 Python 内核。在第 1 节设置 `DATA_DIR`、`GPU` 和运行开关，随后按顺序执行。USB 数据通过 CIL 下载或复用本地数据；预处理和后续计算都在 Notebook 中完成。

如果 Jupyter 中找不到该内核，在已激活的环境中执行：

```bash
python -m ipykernel install --user --name usb-showcase --display-name 'Python (USB showcase)'
```

**验证状态：** 本机已完成 pip 安装、六个 CUDA 扩展编译及 FaCT-GS 训练入口导入检查，`pip check` 通过。USB 读取及预处理实测得到 `(801, 256, 156)`；PyTorch GPU 计算通过。TIGRE 重建测试因其他任务占用 GPU 而退出，完整重建流程尚未验证。
