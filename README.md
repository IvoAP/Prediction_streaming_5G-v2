## Generate Video-Class Datasets with UV

This project converts the `DL_bitrate` column into video quality classes and writes new CSV files for **2, 4, and 6 classes**.

### Run with UV

From the project root, run:

```powershell
uv run -m Scripts.treatment_data
```

Alternative entry point:

```powershell
uv run -m src.main
```

Both commands generate class-based outputs for:

- Amazon Static
- Amazon Driving
- Netflix Static
- Netflix Driving

Output folders are created automatically:

- `Amazon-2/`, `Amazon-4/`, `Amazon-6/`
- `Netflix-2/`, `Netflix-4/`, `Netflix-6/`

Each run prints `class_count` in the terminal for validation.

## Video Class Definition Format

Class assignment is based on `DL_bitrate / 1000` (Mbps), using these bitrate ranges:

- `(0, 0.7)` -> key `0.1`
- `[0.7, 1.1)` -> key `0.7`
- `[1.1, 2.5)` -> key `1.1`
- `[2.5, 5)` -> key `2.5`
- `[5, 20)` -> key `5`
- `[20, +inf)` -> key `20`
- `<= 0` or invalid/missing -> class `0`

### 6-Class Mapping

- `0.1` -> `144p`
- `0.7` -> `360p`
- `1.1` -> `480p`
- `2.5` -> `720p`
- `5` -> `1080p`
- `20` -> `4K`

### 4-Class Mapping

- `0.1` -> `Low`
- `0.7` -> `Low`
- `1.1` -> `HD`
- `2.5` -> `HD`
- `5` -> `1080p`
- `20` -> `4K`

### 2-Class Mapping

- `0.1` -> `Low`
- `0.7` -> `Low`
- `1.1` -> `Low`
- `2.5` -> `High`
- `5` -> `High`
- `20` -> `High`
