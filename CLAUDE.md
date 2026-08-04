This folder contains algorithms for analyzing screen anomalies: black screen, white screen,
screen flickering, screen distortion, and screen freezing, from video files or camera input.

Cross-platform: Linux (V4L2), Windows (MSMF/DSHOW), macOS (AVFoundation).
All platform-specific logic lives in `platform_compat.py` — do not add platform branches
(`sys.platform` checks, hardcoded font/device paths, PowerShell calls) to the analysis scripts.

Setup and troubleshooting: see `README.md`.

Documentation generated from this project should be written as `.md` files into the
directory given by the `CAMERA_ALGO_DOCS_DIR` environment variable; if unset, write to
`docs/` in this repository.
