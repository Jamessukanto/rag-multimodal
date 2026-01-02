"""
Evaluation report generation
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List
from core.config import settings

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate evaluation reports"""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = settings.eval_results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(
        self,
        results: Dict[str, Any],
        filename: str
    ) -> Path:
        """Save results as JSON"""
        file_path = self.results_dir / filename
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON report to {file_path}")
        return file_path
    
    def save_csv(
        self,
        results: Dict[str, Any],
        filename: str
    ) -> Path:
        """Save per-query results as CSV"""
        file_path = self.results_dir / filename
        
        if "per_query" not in results:
            logger.warning("No per-query results to save as CSV")
            return file_path
        
        per_query = results["per_query"]
        if not per_query:
            return file_path
        
        # Get all metric keys
        fieldnames = list(per_query[0].keys())
        
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query)
        
        logger.info(f"Saved CSV report to {file_path}")
        return file_path
    
    def print_console(self, results: Dict[str, Any]):
        """Print results to console"""
        if "aggregated" in results:
            agg = results["aggregated"]
            print("\n=== Evaluation Results ===")
            print(f"Number of queries: {agg['num_queries']}")
            print(f"MRR: {agg['mrr']:.4f}")
            
            for key, value in agg.items():
                if key != "num_queries" and isinstance(value, float):
                    print(f"{key}: {value:.4f}")
        
        if "per_query" in results:
            print(f"\nPer-query results: {len(results['per_query'])} queries")

