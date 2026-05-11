"""Test loading RLDS dataset with both tfds.builder methods."""

import sys

import tensorflow_datasets as tfds

# Test dataset location (created by test_rlds_formatter.py)
data_dir = "/tmp/test_rlds_dataset"
ds_name = "robot_demo"
version = "1.0.1"

print("=" * 60)
print("Testing TFDS Dataset Loading")
print("=" * 60)

# Method 1: Using builder_from_directory
print("\n1. Loading with tfds.builder_from_directory()...")
try:
    builder_dir = f"{data_dir}/{ds_name}"
    builder1 = tfds.builder_from_directory(builder_dir=builder_dir)
    print(f"   ✓ Success! Loaded {builder1.info.name} v{builder1.info.version}")

    # Load dataset
    ds1 = builder1.as_dataset(split="train")
    print("   ✓ Dataset loaded, iterating...")

    for i, example in enumerate(ds1.take(2)):
        print(f"   → Step {i + 1}: observations={list(example['observation'].keys())}")

except Exception as e:
    print(f"   ✗ Failed: {e}")

# Method 2: Using tfds.builder with data_dir
print("\n2. Loading with tfds.builder(name, data_dir, version)...")
try:
    # Add dataset to Python path for import
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)

    # Import the dataset module to register it
    dataset_module = __import__(ds_name)

    # Load with tfds.builder
    builder2 = tfds.builder(ds_name, data_dir=data_dir, version=version)
    print(f"   ✓ Success! Loaded {builder2.info.name} v{builder2.info.version}")

    # Load dataset
    ds2 = builder2.as_dataset(split="train")
    print("   ✓ Dataset loaded, iterating...")

    for i, example in enumerate(ds2.take(2)):
        print(f"   → Step {i + 1}: observations={list(example['observation'].keys())}")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Both methods work! ✓")
print("=" * 60)
