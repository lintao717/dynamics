# Research Data Directory

## Data Acquisition

- **CHECKED** and **CED** are downloaded separately by the researcher from their original sources:
  - CHECKED: https://github.com/cyang03/checked
  - CED (Chinese Rumor Dataset): https://github.com/thunlp/Chinese_Rumor_Dataset
- Raw data remains under `data/raw/` and is **excluded from Git**.
- Only hashed user IDs are processed.
- Exported results contain aggregate statistics and case IDs, not identifiable profiles.

## Directory Structure

```
data/
├── README.md          # This file
├── raw/               # Original datasets (gitignored)
│   ├── CHECKED/       # CHECKED dataset root
│   └── CED/           # CED dataset root
├── processed/         # Derived intermediate files (gitignored)
└── results/           # Replay/calibration outputs (gitignored except schema examples)
```

## Privacy Rules

1. All external user IDs are hashed (SHA-256) before any processing or export.
2. Never attempt identity recovery from hashed IDs.
3. Dataset license/citation and the exact source revision must be recorded in each replay result.
4. Only tiny synthetic or manually minimized fixtures (no real user data) may be committed to `tests/fixtures/`.

## Reproducibility

- Replay results include: git commit hash, parameter version, random seed tuple, and dataset source revision.
- Calibration results include: optimizer settings, bounds, seed tuple, and train/validation split metadata.
