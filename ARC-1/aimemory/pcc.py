from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PCCSchedule:
    restore_pack_order: List[int]

class DeterministicPCC:
    """
    PROFILING: record restore order (pack_idx) during one warmup backward
    EXECUTION: cursor advances, prefetch next lookahead pack indices
    Drift detection:
      - if restored pack_idx != expected schedule[cursor] too often => disable PCC (spill still works)
    """
    def __init__(self, lookahead: int = 4, drift_disable_threshold: int = 64):
        self.lookahead = int(lookahead)
        self.drift_disable_threshold = int(drift_disable_threshold)

        self.mode = "PROFILING"
        self.schedule: Optional[PCCSchedule] = None
        self._record: List[int] = []
        self._cursor: int = -1
        self._drift_count: int = 0
        self.enabled: bool = True

    def reset_step(self):
        self._cursor = -1

    def record_restore(self, pack_idx: int):
        if self.mode == "PROFILING" and self.enabled:
            self._record.append(int(pack_idx))

    def finalize_profile(self):
        if not self._record:
            raise RuntimeError("PCC profiling produced empty schedule (did backward run?)")
        self.schedule = PCCSchedule(restore_pack_order=list(self._record))
        self._record.clear()
        self.mode = "EXECUTION"

    def expected_pack(self) -> Optional[int]:
        if not self.enabled or self.mode != "EXECUTION" or not self.schedule:
            return None
        nxt = self._cursor + 1
        if 0 <= nxt < len(self.schedule.restore_pack_order):
            return self.schedule.restore_pack_order[nxt]
        return None

    def advance_on_restore(self, restored_pack_idx: int):
        if not self.enabled or self.mode != "EXECUTION" or not self.schedule:
            return
        exp = self.expected_pack()
        if exp is not None and int(restored_pack_idx) != int(exp):
            self._drift_count += 1
            if self._drift_count >= self.drift_disable_threshold:
                self.enabled = False
        self._cursor += 1

    def next_prefetch_pack_indices(self) -> List[int]:
        if not self.enabled or self.mode != "EXECUTION" or not self.schedule:
            return []
        start = self._cursor + 1
        end = min(len(self.schedule.restore_pack_order), start + self.lookahead)
        return self.schedule.restore_pack_order[start:end]

    @property
    def drift_count(self) -> int:
        return self._drift_count
