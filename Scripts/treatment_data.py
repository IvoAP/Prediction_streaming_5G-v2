import csv
from collections import Counter
from pathlib import Path


BITRATE_COLUMN = "DL_bitrate"
RESOLUTION_CLASS_COLUMN = "resolution_class"
DROP_COLUMNS = {"PINGAVG", "PINGMIN", "PINGMAX", "PINGSTDEV", "PINGLOSS"}

DATASETS = [
    ("Amazon", "Static", Path("Amazon/Static/amazon_static.csv")),
    ("Amazon", "Driving", Path("Amazon/Driving/amazon_driving.csv")),
    ("Netflix", "Static", Path("Netflix/Static/netflix_static.csv")),
    ("Netflix", "Driving", Path("Netflix/Driving/netflix_driving.csv")),
]

CLASS_OPTIONS = [2, 4, 6]

MISSING_TOKENS = {"", "nan", "none", "null", "na"}

DICTIONARY_RESOLUTION_6_CLASSES = {
    20: "4K",
    5: "1080p",
    2.5: "720p",
    1.1: "480p",
    0.7: "360p",
    0.1: "144p",
}

DICTIONARY_RESOLUTION_4_CLASSES = {
    20: "4K",
    5: "1080p",
    2.5: "HD",
    1.1: "HD",
    0.7: "Low",
    0.1: "Low",
}

DICTIONARY_RESOLUTION_2_CLASSES = {
    20: "High",
    5: "High",
    2.5: "High",
    1.1: "Low",
    0.7: "Low",
    0.1: "Low",
}

CLASS_MAPPINGS = {
    2: DICTIONARY_RESOLUTION_2_CLASSES,
    4: DICTIONARY_RESOLUTION_4_CLASSES,
    6: DICTIONARY_RESOLUTION_6_CLASSES,
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


def _bitrate_bucket_key(value):
    parsed = _parse_numeric(value)
    if parsed is None:
        return None

    value_mbps = parsed / 1000
    if value_mbps <= 0:
        return None
    if value_mbps < 0.7:
        return 0.1
    if value_mbps < 1.1:
        return 0.7
    if value_mbps < 2.5:
        return 1.1
    if value_mbps < 5:
        return 2.5
    if value_mbps < 20:
        return 5
    return 20


def classifier_resolution_range(value, number_of_classes=6):
    if number_of_classes not in CLASS_MAPPINGS:
        raise ValueError("number_of_classes must be one of: 2, 4, 6")

    bucket_key = _bitrate_bucket_key(value)
    if bucket_key is None:
        return 0

    return CLASS_MAPPINGS[number_of_classes][bucket_key]


def process_csv_with_resolution_classes(
    input_csv_path,
    output_csv_path,
    number_of_classes,
    bitrate_column=BITRATE_COLUMN,
    class_column=RESOLUTION_CLASS_COLUMN,
):
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    class_counter = Counter()

    with input_csv_path.open("r", encoding="utf-8", newline="") as src_file:
        reader = csv.DictReader(src_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV without header: {input_csv_path}")

        output_fieldnames = [
            field_name for field_name in reader.fieldnames if field_name not in DROP_COLUMNS
        ]
        if class_column not in output_fieldnames:
            output_fieldnames.append(class_column)

        with output_csv_path.open("w", encoding="utf-8", newline="") as dst_file:
            writer = csv.DictWriter(dst_file, fieldnames=output_fieldnames)
            writer.writeheader()

            for row in reader:
                row = {key: value for key, value in row.items() if key not in DROP_COLUMNS}
                class_value = classifier_resolution_range(
                    row.get(bitrate_column),
                    number_of_classes=number_of_classes,
                )
                row[class_column] = class_value
                class_counter[class_value] += 1
                writer.writerow(row)

    return class_counter


def generate_resolution_datasets():
    print("=== Generating datasets for 2, 4 and 6 classes ===")
    for class_count in CLASS_OPTIONS:
        print(f"\n--- {class_count} classes ---")
        for provider, movement_type, input_path in DATASETS:
            output_dir = Path(f"{provider}-{class_count}") / movement_type
            output_name = f"{provider.lower()}_{class_count}_{movement_type.lower()}.csv"
            output_path = output_dir / output_name

            class_counter = process_csv_with_resolution_classes(
                input_csv_path=input_path,
                output_csv_path=output_path,
                number_of_classes=class_count,
            )

            print(f"{input_path} -> {output_path}")
            print(f"  class_count={dict(class_counter)}")


if __name__ == "__main__":
    generate_resolution_datasets()