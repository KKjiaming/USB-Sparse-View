# Bundled FaCT-GS source

`fact-gs/` contains the tracked FaCT-GS source and all seven populated submodules, as ordinary files, not external Git links. Build artifacts and repository metadata are excluded. Commit IDs and original file hashes are recorded in SOURCES.json.

Local changes: `SceneRecon` accepts an optional `scene_reader`; `optimize` accepts `scene_reader` and `report_callback`. These make USB integration explicit without changing the reconstruction objective or optimizer. Original callers retain their default behavior. The two modified files retain upstream copyright headers.

Licenses are retained in place, including `fact-gs/LICENSE.md`, `fact-gs/fact_gs/r2_gaussian/LICENSE.md`, and each submodule's license. This tree has multiple licenses; the FaCT-GS top-level MIT license does not replace its dependency licenses.

This source tree is required for training and building its CUDA extensions. It contains upstream examples/assets as well as implementation files; the teaching code lives separately under `script/usb_dataset_gaussian/`.
