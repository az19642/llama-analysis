from pathlib import Path

from datasets import concatenate_datasets, load_from_disk

datasets_path = Path("layer_1_split")
dataset_dirs = [
    d for d in datasets_path.iterdir() if d.is_dir() and d.name.startswith("chunk_")
]

# Load all chunk datasets
datasets = [load_from_disk(str(dataset_dir)) for dataset_dir in dataset_dirs]

# Combine all datasets using concatenate_datasets
combined_dataset = concatenate_datasets(datasets)

# Save to disk
combined_dataset.save_to_disk("layer_1_combined")