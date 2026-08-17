## Generate Video-Class Datasets with UV

This project converts the `DL_bitrate` column into video quality classes and writes new CSV files for **6 classes**, keeping only the specified feature columns (`RSRP`, `RSRQ`, `SNR`, `CQI`, `RSSI`).

### Run with UV

From the project root, run:

```powershell
uv run -m Scripts.treatment_data
```

Generates class-based outputs for:

- Amazon Static
- Amazon Driving
- Netflix Static
- Netflix Driving

Output folders are created automatically:

- `Amazon-6/`
- `Netflix-6/`

Each run prints `class_count` in the terminal for validation. Generated CSVs contain only columns: `RSRP,RSRQ,SNR,CQI,RSSI,resolution_class`.

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

