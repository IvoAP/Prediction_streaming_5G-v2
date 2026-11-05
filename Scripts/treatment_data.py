import csv
from collections import Counter
from pathlib import Path

BITRATE_COLUMN = "DL_bitrate"
RESOLUTION_CLASS_COLUMN = "resolution_class"
FEATURE_COLUMNS = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]

DATASETS_DIR = Path("datasets")

DATASETS = [
    ("Amazon", "Static", DATASETS_DIR / "Amazon" / "Static" / "amazon_static.csv"),
    ("Amazon", "Driving", DATASETS_DIR / "Amazon" / "Driving" / "amazon_driving.csv"),
    ("Netflix", "Static", DATASETS_DIR / "Netflix" / "Static" / "netflix_static.csv"),
    ("Netflix", "Driving", DATASETS_DIR / "Netflix" / "Driving" / "netflix_driving.csv"),
]

MISSING_TOKENS = {"", "nan", "none", "null", "na"}

# 6 classes mapping
# <0.7 -> 144p, 0.7-1.1 -> 360p, 1.1-2.5 -> 480p, 2.5-5.0 -> 720p, 5.0-20.0 -> 1080p, >=20.0 -> 4K
DICTIONARY_RESOLUTION_6_CLASSES = {
    20: "4K",
    5: "1080p",
    2.5: "720p",
    1.1: "480p",
    0.7: "360p",
    0.1: "144p",
}

# 4 classes mapping
# <1.1 -> SD (144p + 360p)
# 1.1-5.0 -> HD (480p + 720p)
# 5.0-20.0 -> FHD (1080p)
# >=20.0 -> 4K (4K)
DICTIONARY_RESOLUTION_4_CLASSES = {
    20: "4K",
    5: "FHD",
    1.1: "HD",
    0.1: "SD",
}

# 2 classes mapping
# <2.5 -> SD (<720p)
# >=2.5 -> HD (>=720p)
DICTIONARY_RESOLUTION_2_CLASSES = {
    2.5: "HD",
    0.1: "SD",
}


def _normalize_value(value):
    if value is None:
        return ""
    return str(value).strip()


def _parse_numeric(value):
    normalized = _normalize_value(value)
    if normalized.lower() in MISSING_TOKENS or normalized == "-":
        return None

    normalized = normalized.replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def get_resolution_class(value, num_classes=6):
    parsed = _parse_numeric(value)
    if parsed is None:
        return 0

    value_mbps = parsed / 1000
    if value_mbps <= 0:
        return 0

    if num_classes == 6:
        if value_mbps < 0.7:
            return "144p"
        if value_mbps < 1.1:
            return "360p"
        if value_mbps < 2.5:
            return "480p"
        if value_mbps < 5.0:
            return "720p"
        if value_mbps < 20.0:
            return "1080p"
        return "4K"

    elif num_classes == 4:
        if value_mbps < 1.1:
            return "SD"
        if value_mbps < 5.0:
            return "HD"
        if value_mbps < 20.0:
            return "FHD"
        return "4K"

    elif num_classes == 2:
        if value_mbps < 2.5:
            return "SD"
        return "HD"

    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}. Choose from [2, 4, 6].")


def process_csv_with_resolution_classes(
    input_csv_path,
    output_csv_path,
    num_classes=6,
    bitrate_column=BITRATE_COLUMN,
    class_column=RESOLUTION_CLASS_COLUMN,
    feature_columns=FEATURE_COLUMNS,
):
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    class_counter = Counter()

    output_fieldnames = list(feature_columns)
    if class_column not in output_fieldnames:
        output_fieldnames.append(class_column)

    with input_csv_path.open("r", encoding="utf-8", newline="") as src_file:
        reader = csv.DictReader(src_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV without header: {input_csv_path}")

        with output_csv_path.open("w", encoding="utf-8", newline="") as dst_file:
            writer = csv.DictWriter(dst_file, fieldnames=output_fieldnames)
            writer.writeheader()

            for row in reader:
                class_value = get_resolution_class(row.get(bitrate_column), num_classes=num_classes)
                class_counter[class_value] += 1

                filtered_row = {col: row.get(col, "") for col in feature_columns}
                filtered_row[class_column] = class_value
                writer.writerow(filtered_row)

    return class_counter


def generate_all_resolution_datasets(target_classes=(2, 4, 6)):
    for num_classes in target_classes:
        print(f"\n=== Generating datasets for {num_classes} classes ===")
        for provider, movement_type, input_path in DATASETS:
            output_dir = DATASETS_DIR / f"{provider}-{num_classes}" / movement_type
            output_name = f"{provider.lower()}_{num_classes}_{movement_type.lower()}.csv"
            output_path = output_dir / output_name

            class_counter = process_csv_with_resolution_classes(
                input_csv_path=input_path,
                output_csv_path=output_path,
                num_classes=num_classes,
            )

            print(f"{input_path} -> {output_path}")
            print(f"  class_count={dict(class_counter)}")


if __name__ == "__main__":
    generate_all_resolution_datasets()
