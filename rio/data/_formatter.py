# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Base formatter class for converting robodm data to other formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger


class Formatter(ABC):
    """
    Abstract base class for converting robodm trajectory files to other dataset formats.

    This class provides a common interface for loading robodm data and converting it
    to various target formats (e.g., LeRobot, RLDS, etc.).
    """

    def __init__(self, robodm_path: str | Path, output_path: str | Path, verbose: bool = False):
        """
        Initialize the formatter.

        Args:
            robodm_path: Path to the robodm trajectory file (.vla) or directory containing multiple .vla files
            output_path: Path where the converted dataset should be saved
            verbose: If True, print progress information
        """
        self.robodm_path = Path(robodm_path)
        self.output_path = Path(output_path)
        self.verbose = verbose

        if not self.robodm_path.exists():
            raise FileNotFoundError(f"Path not found: {self.robodm_path}")

        # Configure logger level based on verbose flag
        if not verbose:
            logger.disable("rio.data.formatter")

    def convert(self):
        """
        Main conversion pipeline.

        This method orchestrates the conversion process:
        1. Process and transform the data
        2. Write to target format

        Subclasses handle loading robodm data as needed.
        """
        logger.debug(f"Converting from {self.robodm_path} to {self.__class__.__name__} format")

        converted_data = self._process_data()

        logger.debug(f"Writing output to {self.output_path}")

        self._write_output(converted_data)

        logger.debug("Conversion complete!")

    @abstractmethod
    def _process_data(self) -> Any:
        """
        Process and transform robodm data to target format.

        This method should be implemented by subclasses to handle
        format-specific data transformations, including loading
        trajectory files from self.robodm_path.

        Returns:
            Processed data in a format suitable for the target dataset
        """
        pass

    @abstractmethod
    def _write_output(self, data: Any):
        """
        Write the converted data to disk.

        This method should be implemented by subclasses to handle
        format-specific file writing.

        Args:
            data: Processed data ready to be written
        """
        pass
