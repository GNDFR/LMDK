# lmdk/dpe.py
"""
Data Pipeline Engine (DPE) - Python wrapper for Rust data processing modules
"""

from .rust_core import DataCleanser
from typing import List, Optional
import os

class DataPipeline:
    """
    High-level interface for data preprocessing pipeline.
    """

    def __init__(self, min_length: int = 20, toxic_keywords: Optional[List[str]] = None):
        self.cleanser = DataCleanser(min_length=min_length, toxic_keywords=toxic_keywords)

    def process_file(self, filepath: str) -> int:
        """
        Process a text file and return the number of unique processed lines.
        """
        return self.cleanser.process_file(filepath)

    def get_unique_count(self) -> int:
        """
        Get the count of unique processed lines.
        """
        return self.cleanser.count

    def save_processed_data(self, output_path: str):
        """
        Placeholder for saving processed data (to be implemented).
        """
        # Future: save the processed unique lines to a file
        pass