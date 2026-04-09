from dataclasses import dataclass, field
import threading


@dataclass
class State:
    total_reps: int = 0
    correct_reps: int = 0
    incorrect_reps: int = 0
    last_prediction: str | None = None
    last_confidence: float | None = None
    history: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self):
        with self.lock:
            return {
                "reps": self.total_reps,
                "correct_reps": self.correct_reps,
                "incorrect_reps": self.incorrect_reps,
                "status": self.last_prediction,
                "confidence": self.last_confidence,
                "history": list(self.history),
            }