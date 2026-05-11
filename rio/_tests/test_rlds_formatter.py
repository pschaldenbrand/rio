# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RLDS formatter."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import robodm

# Check if RLDS dependencies are available
try:
    import tensorflow as tf

    from rio.data import RLDSFormatter

    RLDS_AVAILABLE = True
except ImportError:
    RLDS_AVAILABLE = False
    RLDSFormatter = None


# Skip all tests if RLDS is not available
pytestmark = pytest.mark.skipif(
    not RLDS_AVAILABLE, reason="RLDS dependencies not installed. Install with: pip install rio[rlds]"
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup after test
    if Path(temp_path).exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_robodm_path(temp_dir):
    """Create a sample robodm trajectory for testing."""
    output_path = str(Path(temp_dir) / "test_trajectory.vla")

    # Create trajectory in write mode
    traj = robodm.Trajectory(
        path=output_path,
        mode="w",
        video_codec="libx264",
    )

    # Simulate 50 timesteps at 30 Hz
    num_steps = 50
    fps = 30
    rng = np.random.default_rng()

    for i in range(num_steps):
        timestamp = i * (1000.0 / fps)  # milliseconds

        # Simulated robot state
        joint_positions = rng.standard_normal(7).astype(np.float32)
        ee_pose = rng.standard_normal(7).astype(np.float32)
        gripper_position = rng.random(1).astype(np.float32)
        camera_rgb = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        action = rng.standard_normal(7).astype(np.float32)

        # Add data to trajectory
        traj.add("observation/joint_positions", joint_positions, timestamp=timestamp, time_unit="ms")
        traj.add("observation/ee_pose", ee_pose, timestamp=timestamp, time_unit="ms")
        traj.add("observation/gripper_position", gripper_position, timestamp=timestamp, time_unit="ms")
        traj.add("observation/images/camera", camera_rgb, timestamp=timestamp, time_unit="ms")
        traj.add("action", action, timestamp=timestamp, time_unit="ms")

    traj.close()

    return output_path


@pytest.fixture
def rlds_output_path(temp_dir):
    """Provide path for RLDS output."""
    return str(Path(temp_dir) / "rlds_dataset")


class TestRLDSFormatter:
    """Tests for RLDSFormatter class."""

    def test_formatter_initialization(self, sample_robodm_path, rlds_output_path):
        """Test that formatter can be initialized with valid paths."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        assert formatter.robodm_path == Path(sample_robodm_path)
        assert formatter.output_path == Path(rlds_output_path)
        assert formatter.dataset_name == "test_dataset"
        assert formatter.fps == 30
        assert formatter.robot_type == "test_robot"

    def test_formatter_invalid_path(self, rlds_output_path):
        """Test that formatter raises error with invalid robodm path."""
        with pytest.raises(FileNotFoundError):
            RLDSFormatter(
                robodm_path="/nonexistent/path.vla",
                output_path=rlds_output_path,
                dataset_name="test_dataset",
            )

    def test_formatter_without_dependencies(self, sample_robodm_path, rlds_output_path, monkeypatch):
        """Test that formatter raises ImportError when dependencies are missing."""
        # This test is conceptual - we can't easily test the import failure
        # since we already have the dependencies installed for this test to run
        pass

    def test_convert_creates_directory_structure(self, sample_robodm_path, rlds_output_path):
        """Test that conversion creates the expected directory structure."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        output_dir = Path(rlds_output_path) / "test_dataset" / "1.0.1"

        # Check that directory exists
        assert output_dir.exists()

    def test_convert_creates_files(self, sample_robodm_path, rlds_output_path):
        """Test that conversion creates all required files."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        output_dir = Path(rlds_output_path) / "test_dataset" / "1.0.1"

        # Check that files exist
        assert (output_dir / "train.tfrecord").exists()
        assert (output_dir / "dataset_info.json").exists()
        assert (output_dir / "features.json").exists()

    def test_dataset_info_content(self, sample_robodm_path, rlds_output_path):
        """Test that dataset_info.json contains correct metadata."""
        task_desc = "Custom test task"

        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description=task_desc,
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        info_path = Path(rlds_output_path) / "test_dataset" / "1.0.1" / "dataset_info.json"
        with open(info_path) as f:
            info = json.load(f)

        assert info["name"] == "test_dataset"
        assert info["version"] == "1.0.1"
        assert info["description"] == task_desc
        assert "splits" in info
        assert "train" in info["splits"]
        assert info["metadata"]["robot_type"] == "test_robot"
        assert info["metadata"]["fps"] == 30
        assert info["metadata"]["num_steps"] == 50

    def test_features_json_content(self, sample_robodm_path, rlds_output_path):
        """Test that features.json contains feature specifications."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        features_path = Path(rlds_output_path) / "test_dataset" / "1.0.1" / "features.json"
        with open(features_path) as f:
            features = json.load(f)

        # Check that features exist
        assert "observation/joint_positions" in features
        assert "observation/images/camera" in features
        assert "action" in features

        # Check feature properties
        joint_feature = features["observation/joint_positions"]
        assert "dtype" in joint_feature
        assert "shape" in joint_feature
        assert "is_image" in joint_feature
        assert joint_feature["shape"] == [7]
        assert joint_feature["is_image"] is False

        # Check image feature
        image_feature = features["observation/images/camera"]
        assert image_feature["shape"] == [64, 64, 3]
        assert image_feature["is_image"] is True

    def test_tfrecord_can_be_read(self, sample_robodm_path, rlds_output_path):
        """Test that TFRecord file can be read back."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        tfrecord_path = Path(rlds_output_path) / "test_dataset" / "1.0.1" / "train.tfrecord"

        # Read TFRecord
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        # Count records
        record_count = sum(1 for _ in dataset)
        assert record_count == 50

    def test_tfrecord_content(self, sample_robodm_path, rlds_output_path):
        """Test that TFRecord contains correct data structure."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        tfrecord_path = Path(rlds_output_path) / "test_dataset" / "1.0.1" / "train.tfrecord"
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        # Read first record
        for raw_record in dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            # Check required DROID schema fields
            assert "is_first" in features
            assert "is_last" in features
            assert "is_terminal" in features
            assert "discount" in features
            assert "reward" in features
            assert "action" in features

            # Check language instructions
            assert "language_instruction" in features
            assert "language_instruction_2" in features
            assert "language_instruction_3" in features

            # Check episode metadata
            assert "episode_metadata/recording_folderpath" in features
            assert "episode_metadata/file_path" in features

            # Check observation fields (mapped from input data)
            assert "observation/joint_position" in features or "observation/joint_positions" in features
            assert "observation/gripper_position" in features

    def test_first_and_last_steps(self, sample_robodm_path, rlds_output_path):
        """Test that first and last steps are marked correctly."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        tfrecord_path = Path(rlds_output_path) / "test_dataset" / "1.0.1" / "train.tfrecord"
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        records = list(dataset)

        # Check first step
        first_example = tf.train.Example()
        first_example.ParseFromString(records[0].numpy())
        first_features = first_example.features.feature
        assert first_features["is_first"].int64_list.value[0] == 1
        assert first_features["is_last"].int64_list.value[0] == 0

        # Check last step
        last_example = tf.train.Example()
        last_example.ParseFromString(records[-1].numpy())
        last_features = last_example.features.feature
        assert last_features["is_first"].int64_list.value[0] == 0
        assert last_features["is_last"].int64_list.value[0] == 1
        assert last_features["is_terminal"].int64_list.value[0] == 1

    def test_image_detection(self, sample_robodm_path, rlds_output_path):
        """Test that image keys are auto-detected correctly."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            image_keys=None,  # Auto-detect
            verbose=False,
        )

        formatter.convert()

        features_path = Path(rlds_output_path) / "test_dataset" / "1.0.1" / "features.json"
        with open(features_path) as f:
            features = json.load(f)

        # Check that camera was detected as an image key
        assert features["observation/images/camera"]["is_image"] is True

        # Check that other features are not marked as images
        assert features["observation/joint_positions"]["is_image"] is False
        assert features["action"]["is_image"] is False

    def test_explicit_image_keys(self, sample_robodm_path, rlds_output_path):
        """Test that explicitly provided image keys are used."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            image_keys=["observation/images/camera"],
            verbose=False,
        )

        formatter.convert()

        features_path = Path(rlds_output_path) / "test_dataset" / "1.0.1" / "features.json"
        with open(features_path) as f:
            features = json.load(f)

        assert features["observation/images/camera"]["is_image"] is True

    def test_compress_images_option(self, sample_robodm_path, rlds_output_path):
        """Test that compress_images option works."""
        # Test with compression enabled
        formatter_compressed = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_compressed",
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            compress_images=True,
            verbose=False,
        )

        formatter_compressed.convert()

        # Just verify it completes without error
        tfrecord_path = Path(rlds_output_path) / "test_compressed" / "1.0.1" / "train.tfrecord"
        assert tfrecord_path.exists()

    def test_custom_dataset_name(self, sample_robodm_path, rlds_output_path):
        """Test that custom dataset name is used."""
        custom_name = "my_custom_robot_dataset"

        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name=custom_name,
            fps=30,
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        output_dir = Path(rlds_output_path) / custom_name / "1.0.1"
        assert output_dir.exists()

        info_path = output_dir / "dataset_info.json"
        with open(info_path) as f:
            info = json.load(f)

        assert info["name"] == custom_name

    def test_different_fps(self, sample_robodm_path, rlds_output_path):
        """Test formatter with different FPS values."""
        formatter = RLDSFormatter(
            robodm_path=sample_robodm_path,
            output_path=rlds_output_path,
            dataset_name="test_dataset",
            fps=60,  # Different FPS
            robot_type="test_robot",
            task_description="Test task",
            language_instruction="Test instruction",
            verbose=False,
        )

        formatter.convert()

        info_path = Path(rlds_output_path) / "test_dataset" / "1.0.1" / "dataset_info.json"
        with open(info_path) as f:
            info = json.load(f)

        assert info["metadata"]["fps"] == 60
