from dataclasses import dataclass, field
from threading import Lock


@dataclass
class LiveState:
    reps: int = 0
    status: str = "Waiting"

    lock: Lock = field(default_factory=Lock)

    def set_status(self, status: str):
        with self.lock:
            self.status = status

    def increment_rep(self):
        with self.lock:
            self.reps += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "reps": self.reps,
                "status": self.status,
            }