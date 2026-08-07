from datetime import datetime


def get_recent_window(timestamps: list[datetime], window_size: int) -> list[int]:
    if len(timestamps) < window_size:
        raise ValueError(
            f"Need at least {window_size} timestamps, got {len(timestamps)}"
        )
    return list(range(len(timestamps) - window_size, len(timestamps)))
