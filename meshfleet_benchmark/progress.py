"""
Progress tracking functionality for the MeshFleet benchmark pipeline.
"""

import time
import logging
from typing import List


class ProgressTracker:
    """Tracks and displays progress across pipeline stages."""
    
    def __init__(self, stages: List[str]):
        self.stages = stages
        self.current_stage_idx = 0
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.completed_stages = []
    
    def start_stage(self, stage_name: str) -> None:
        """Start a new stage."""
        if stage_name in self.stages:
            self.current_stage_idx = self.stages.index(stage_name)
        
        self.stage_start_time = time.time()
        elapsed = time.time() - self.start_time
        
        logging.info(f"{'='*60}")
        logging.info(f"Starting stage: {stage_name}")
        logging.info(f"Progress: {len(self.completed_stages)}/{len(self.stages)} stages completed")
        logging.info(f"Total elapsed time: {elapsed:.1f}s")
        logging.info(f"{'='*60}")
    
    def complete_stage(self, stage_name: str) -> None:
        """Complete the current stage."""
        stage_duration = time.time() - self.stage_start_time
        total_elapsed = time.time() - self.start_time
        
        self.completed_stages.append(stage_name)
        
        # Estimate remaining time
        if len(self.completed_stages) > 0:
            avg_stage_time = total_elapsed / len(self.completed_stages)
            remaining_stages = len(self.stages) - len(self.completed_stages)
            estimated_remaining = avg_stage_time * remaining_stages
        else:
            estimated_remaining = 0
        
        logging.info(f"Completed stage: {stage_name} ({stage_duration:.1f}s)")
        logging.info(f"Estimated time remaining: {estimated_remaining:.1f}s")
    
    def get_progress_summary(self) -> str:
        """Get a summary of progress."""
        total_elapsed = time.time() - self.start_time
        completed = len(self.completed_stages)
        total = len(self.stages)
        return f"Progress: {completed}/{total} stages, {total_elapsed:.1f}s elapsed"