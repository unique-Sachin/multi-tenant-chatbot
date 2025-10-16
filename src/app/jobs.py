"""Background job system for running ingestion tasks."""

import threading
import traceback
from typing import Dict, Callable, Any, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobManager:
    """Manages background jobs with thread-based execution."""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def start_job(
        self,
        job_id: str,
        target: Callable,
        args: tuple = (),
        kwargs: dict = None
    ) -> bool:
        """Start a background job.
        
        Args:
            job_id: Unique identifier for the job
            target: Function to execute
            args: Positional arguments for target
            kwargs: Keyword arguments for target
            
        Returns:
            True if job started successfully
        """
        if kwargs is None:
            kwargs = {}
        
        with self.lock:
            if job_id in self.jobs and self.jobs[job_id]["status"] == "running":
                logger.warning(f"Job {job_id} is already running")
                return False
            
            # Create job entry
            self.jobs[job_id] = {
                "status": "running",
                "started_at": datetime.now(),
                "completed_at": None,
                "error": None,
                "result": None
            }
        
        # Start thread
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, target, args, kwargs),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started job {job_id}")
        return True
    
    def _run_job(
        self,
        job_id: str,
        target: Callable,
        args: tuple,
        kwargs: dict
    ):
        """Internal method to run job and update status."""
        try:
            logger.info(f"Executing job {job_id}")
            result = target(*args, **kwargs)
            
            with self.lock:
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["completed_at"] = datetime.now()
                self.jobs[job_id]["result"] = result
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Job {job_id} failed: {error_msg}")
            
            with self.lock:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["completed_at"] = datetime.now()
                self.jobs[job_id]["error"] = error_msg
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        """List all jobs."""
        with self.lock:
            return self.jobs.copy()


# Global job manager instance
job_manager = JobManager()
