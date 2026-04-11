from dataclasses import dataclass

@dataclass
class Counter:
    phase: str = "top"

    def update(self, rep_phase: str) -> bool:
        """
        Returns True when a full rep is completed:
        top -> bottom -> top
        """
        if rep_phase == "unknown":
            return False

        if self.phase == "top" and rep_phase == "bottom":
            self.phase = "bottom"
            return False

        if self.phase == "bottom" and rep_phase == "top":
            self.phase = "top"
            return True

        return False

    def reset(self) -> None:
        self.phase = "top"