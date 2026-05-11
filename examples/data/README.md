# RoboDM Dataset Formatters

This directory contains tools and examples for converting robodm trajectory files to standard dataset formats like LeRobot and RLDS.

## Command-Line Conversion Tool

The easiest way to convert datasets is using the `convert.py` script:

```bash
# Convert to LeRobot format (default)
python examples/data/convert.py

# Convert with custom input path and task
python examples/data/convert.py -i /tmp/my_demo.vla -t "pick and place" -v

# Convert to RLDS format
python examples/data/convert.py -f rlds

# Clean output directory before conversion
python examples/data/convert.py --clean -v

# See all options
python examples/data/convert.py --help
```

**Default settings:**
- Input: `/tmp/collected_data/*.vla` (supports glob patterns)
- Format: `lerobot`
- Task: `"make circle"`
- FPS: `30`

## Supported Formats

### LeRobot Format

LeRobot is a format designed for robot learning datasets with efficient storage and access patterns.

**Quick Start (Python API):**

```python
from rio.data import LeRobotFormatter

# Convert a robodm trajectory to LeRobot format
formatter = LeRobotFormatter(
    robodm_path="/tmp/demo.vla",
    output_path="/tmp/lerobot_dataset",
    repo_id="user/my_robot_dataset",
    fps=30,
    robot_type="franka",
    task="Pick and place objects",
    verbose=True
)

formatter.convert()
```

**Running the test example:**

```bash
python examples/data/test_lerobot_formatter.py
```

This will:
1. Create a sample robodm trajectory with simulated robot data
2. Convert it to LeRobot format
3. Verify the output structure

### RLDS Format

RLDS (Robot Learning Dataset) is a TensorFlow-based format for robot learning datasets.

**Quick Start (Python API):**

```python
from rio.data import RLDSFormatter

# Convert a robodm trajectory to RLDS format
formatter = RLDSFormatter(
    robodm_path="/tmp/demo.vla",
    output_path="/tmp/rlds_dataset",
    dataset_name="robot_demo",
    fps=30,
    robot_type="franka",
    task_description="Pick and place objects",
    compress_images=True,
    verbose=True
)

formatter.convert()
```

**Requirements:**

RLDS format requires additional dependencies:
```bash
pip install rio[rlds]
# or
uv pip install tensorflow tensorflow-datasets
```

**Running the test example:**

```bash
python examples/data/test_rlds_formatter.py
```


## Output Structures

### LeRobot Output Structure

The formatter creates a LeRobot-compatible dataset with the following structure:

```
output_path/
├── data/
│   └── chunk-000/
│       └── file-000.parquet         # Timestamped observations and actions
├── meta/
│   ├── info.json                    # Dataset metadata (fps, features, shapes)
│   ├── stats.json                   # Feature statistics for normalization
│   ├── tasks.parquet                # Task descriptions
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet     # Episode metadata
└── videos/                          # Video files (requires additional setup)
    └── <feature_name>/
        └── chunk-000/
            └── file-000.mp4
```


### RLDS Output Structure

The RLDS formatter creates a TensorFlow Datasets-compatible structure:

```
output_path/
└── <dataset_name>/
    └── 1.0.0/
        ├── train.tfrecord           # TFRecord with episode data
        ├── dataset_info.json        # Dataset metadata
        └── features.json            # Feature specifications
```

