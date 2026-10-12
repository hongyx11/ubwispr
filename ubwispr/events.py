from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Status:
    state: str
    message: str
    notify: bool = False


StatusCallback = Callable[[Status], None]
