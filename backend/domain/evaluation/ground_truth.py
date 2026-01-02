"""
Ground truth management
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)


class GroundTruthManager:
    """Manages ground truth data"""
    
    def __init__(self, ground_truth_file: Optional[Path] = None):
        self.ground_truth_file = ground_truth_file
        self._ground_truth: Dict[str, Set[str]] = {}
        if ground_truth_file and ground_truth_file.exists():
            self.load(ground_truth_file)
    
    def load(self, file_path: Path):
        """Load ground truth from file"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                self._ground_truth = {
                    query: set(relevant) 
                    for query, relevant in data.items()
                }
            logger.info(f"Loaded ground truth from {file_path}")
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            raise
    
    def save(self, file_path: Path):
        """Save ground truth to file"""
        try:
            data = {
                query: list(relevant)
                for query, relevant in self._ground_truth.items()
            }
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved ground truth to {file_path}")
        except Exception as e:
            logger.error(f"Error saving ground truth: {e}")
            raise
    
    def get_relevant(self, query: str) -> Set[str]:
        """Get relevant document IDs for a query"""
        return self._ground_truth.get(query, set())
    
    def add(self, query: str, relevant_doc_ids: List[str]):
        """Add ground truth for a query"""
        self._ground_truth[query] = set(relevant_doc_ids)
    
    def has_ground_truth(self, query: str) -> bool:
        """Check if ground truth exists for a query"""
        return query in self._ground_truth

