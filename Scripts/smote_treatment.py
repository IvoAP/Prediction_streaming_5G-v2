import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


BITRATE_COLUMN = "DL_bitrate"
RESOLUTION_CLASS_COLUMN = "resolution_class"
FEATURE_COLUMNS = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
K_NEIGHBORS = 5
RANDOM_SEED = 42

DATASETS = [
    ("Amazon", "Static", Path("Amazon-6/Static/amazon_6_static.csv")),
    ("Amazon", "Driving", Path("Amazon-6/Driving/amazon_6_driving.csv")),
    ("Netflix", "Static", Path("Netflix-6/Static/netflix_6_static.csv")),
    ("Netflix", "Driving", Path("Netflix-6/Driving/netflix_6_driving.csv")),
]

MISSING_TOKENS = {"", "nan", "none", "null", "na"}

DICTIONARY_RESOLUTION_6_CLASSES = {
    20: "4K",
    5: "1080p",
    2.5: "720p",
    1.1: "480p",
    0.7: "360p",
    0.1: "144p",
}


def read_csv_dataset(input_csv_path):
    rows = []

    with input_csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV without header: {input_csv_path}")

        for row in reader:
            rows.append(row)

    return rows


def parse_feature_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_to_sample(row):
    features = []

    for column in FEATURE_COLUMNS:
        value = parse_feature_value(row.get(column))

        if value is None:
            return None

        features.append(value)

    return {
        "features": features,
        "class": row.get(RESOLUTION_CLASS_COLUMN),
    }


def sample_to_row(sample):
    row = {}

    for index, column in enumerate(FEATURE_COLUMNS):
        row[column] = round(sample["features"][index], 2)

    row[RESOLUTION_CLASS_COLUMN] = sample["class"]

    return row


def group_samples_by_class(rows):
    samples_by_class = defaultdict(list)

    for row in rows:
        sample = row_to_sample(row)

        if sample is None:
            continue

        samples_by_class[sample["class"]].append(sample)

    return samples_by_class


def count_classes(samples_by_class):
    return Counter(
        {
            class_name: len(samples) for class_name, samples in samples_by_class.items()
        }
    )


def euclidean_distance(sample_a, sample_b):
    squared_distance = 0.0

    for value_a, value_b in zip(
        sample_a["features"],
        sample_b["features"]
    ):
        difference = value_a - value_b
        squared_distance += difference ** 2

    return math.sqrt(squared_distance)


def get_neighbors(sample, class_samples, k):
    distances = []

    for other_sample in class_samples:
        if other_sample is sample:
            continue

        distance = euclidean_distance(sample, other_sample)

        distances.append((distance, other_sample))

    distances.sort(key=lambda item: item[0])

    neighbors = [
        sample_data
        for _, sample_data in distances[:k]
    ]

    return neighbors


def generate_synthetic_sample(sample, neighbor):
    synthetic_features = []

    interpolation_factor = random.random()

    for value, neighbor_value in zip(
        sample["features"],
        neighbor["features"]
    ):
        difference = neighbor_value - value

        synthetic_value = (
            value + interpolation_factor * difference
        )

        synthetic_features.append(synthetic_value)

    return {
        "features": synthetic_features,
        "class": sample["class"],
    }


def write_csv_dataset(output_csv_path, samples):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FEATURE_COLUMNS + [RESOLUTION_CLASS_COLUMN],
        )

        writer.writeheader()

        for sample in samples:
            writer.writerow(sample_to_row(sample))


def process_dataset(input_csv_path, output_csv_path):
    rows = read_csv_dataset(input_csv_path)

    samples_by_class = group_samples_by_class(rows)

    original_counts = count_classes(samples_by_class)

    balanced_samples = apply_smote(samples_by_class)

    balanced_by_class = defaultdict(list)

    for sample in balanced_samples:
        balanced_by_class[sample["class"]].append(sample)

    balanced_counts = count_classes(balanced_by_class)

    write_csv_dataset(
        output_csv_path,
        balanced_samples,
    )

    return original_counts, balanced_counts


def apply_smote(samples_by_class):
    class_counts = count_classes(samples_by_class)
    target_count = max(class_counts.values())

    balanced_samples = []

    for class_name, class_samples in samples_by_class.items():
        balanced_samples.extend(class_samples)

        samples_to_generate = target_count - len(class_samples)

        if samples_to_generate <= 0:
            continue

        if len(class_samples) < 2:
            raise ValueError(
                f"Cannot apply SMOTE to class '{class_name}' "
                "with fewer than 2 samples."
            )

        k = min(K_NEIGHBORS, len(class_samples) - 1)

        for _ in range(samples_to_generate):
            sample = random.choice(class_samples)

            neighbors = get_neighbors(
                sample,
                class_samples,
                k,
            )

            neighbor = random.choice(neighbors)

            synthetic_sample = generate_synthetic_sample(
                sample,
                neighbor,
            )

            balanced_samples.append(synthetic_sample)

    return balanced_samples


def generate_smote_datasets():
    print("=== Generating SMOTE datasets ===")

    random.seed(RANDOM_SEED)

    for provider, movement_type, input_path in DATASETS:
        output_dir = Path(
            f"{provider}-6-SMOTE"
        ) / movement_type

        output_name = (
            f"{provider.lower()}_6_"
            f"{movement_type.lower()}_smote.csv"
        )

        output_path = output_dir / output_name

        original_counts, balanced_counts = process_dataset(
            input_csv_path=input_path,
            output_csv_path=output_path,
        )

        print(f"{input_path} -> {output_path}")

        print(
            f"  before={dict(original_counts)}"
        )

        print(
            f"  after={dict(balanced_counts)}"
        )

if __name__ == "__main__":
    generate_smote_datasets()