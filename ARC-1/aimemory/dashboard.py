def Dashboard(*args, **kwargs):
    raise RuntimeError(
        "Dashboard requires extras. Install with: pip install 'arc1[dashboard]'"
    )

try:
    from rich.live import Live
    from rich.table import Table
    import time
    import pynvml

    class Dashboard:  # type: ignore
        def __init__(self, ctrl, title="ARC-1"):
            self.ctrl = ctrl
            self.title = title
            pynvml.nvmlInit()
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

        def _table(self):
            m = self.ctrl.metrics
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)
            used = info.used / 1e9
            total = info.total / 1e9

            t = Table(title=self.title, style="cyan")
            t.add_column("Metric")
            t.add_column("Value")
            t.add_column("Status")

            t.add_row("Backend", str(m.backend), "[blue]MODE[/]")
            t.add_row("VRAM", f"{used:.2f}/{total:.2f} GB", "[green]OK[/]")
            t.add_row("Safe Mode", str(m.safe_mode), "[red]ON[/]" if m.safe_mode else "[green]OFF[/]")

            t.add_row("Prefetch Hit Rate", f"{m.prefetch_hit_rate():.2f}%",
                      "[green]GOOD[/]" if m.prefetch_hit_rate() > 85 else "[yellow]TUNE[/]")
            t.add_row("PCC Enabled", str(m.pcc_enabled), "[green]ON[/]" if m.pcc_enabled else "[yellow]OFF[/]")
            t.add_row("PCC Drift", str(m.pcc_drift_count), "[blue]DRIFT[/]")

            return t

        def start(self):
            with Live(self._table(), refresh_per_second=4) as live:
                while True:
                    live.update(self._table())
                    time.sleep(0.25)

except Exception:
    pass
