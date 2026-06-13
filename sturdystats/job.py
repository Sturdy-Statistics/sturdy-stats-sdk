import time

from .base import SturdyStatsSdkError


_TERMINAL_OK = {"succeeded"}
_TERMINAL_FAIL = {"failed", "cancelled"}
_TERMINAL = _TERMINAL_OK | _TERMINAL_FAIL


class Job:
    """Represents an in-flight async job. Call wait() to block until complete.
    → GET /jobs/{job-id}
    """

    def __init__(self, base, job_id: str):
        self._base = base
        self.job_id = job_id

    def status(self) -> dict:
        """Fetch current job state.
        → GET /jobs/{job-id}
        """
        return self._base._get(f"jobs/{self.job_id}")

    def wait(self, poll_interval_start: float = 1.0, poll_interval_max: float = 60.0) -> dict:
        """Block until job reaches a terminal state. Returns final job dict.
        Raises SturdyStatsSdkError if the job fails or is cancelled.
        → GET /jobs/{job-id}
        """
        interval = poll_interval_start
        while True:
            data = self.status()
            s = data.get("status")
            if s in _TERMINAL_OK:
                return data
            if s in _TERMINAL_FAIL:
                raise SturdyStatsSdkError(0, f"Job {self.job_id} ended with status '{s}': {data}")
            time.sleep(interval)
            interval = min(interval * 1.5, poll_interval_max)

    def __repr__(self):
        return f"Job(id={self.job_id!r})"
