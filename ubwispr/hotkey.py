import logging
import threading
import time
from typing import Callable

from pynput import keyboard

log = logging.getLogger(__name__)


_KEY_MAP: dict[str, keyboard.Key] = {
    "alt_r": keyboard.Key.alt_r,
    "alt_l": keyboard.Key.alt_l,
    "alt_gr": keyboard.Key.alt_gr,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl_l": keyboard.Key.ctrl_l,
    "cmd_r": keyboard.Key.cmd_r,
    "cmd_l": keyboard.Key.cmd_l,
    "shift_r": keyboard.Key.shift_r,
    "shift_l": keyboard.Key.shift_l,
    "f1": keyboard.Key.f1, "f2": keyboard.Key.f2, "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4, "f5": keyboard.Key.f5, "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7, "f8": keyboard.Key.f8, "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10, "f11": keyboard.Key.f11, "f12": keyboard.Key.f12,
}


def resolve_key(name: str) -> keyboard.Key:
    key = _KEY_MAP.get(name)
    if key is None:
        raise ValueError(f"Unsupported hotkey {name!r}; supported: {sorted(_KEY_MAP)}")
    return key


class HoldHotkey:
    """Run a global press/release listener on a single modifier key.

    Debounces OS key-repeat so the press/release callbacks only fire on
    actual down/up edges.
    """

    def __init__(
        self,
        key_name: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ):
        self.key = resolve_key(key_name)
        self._on_press = on_press
        self._on_release = on_release
        self._held = False
        self._lock = threading.Lock()
        self._released = threading.Event()
        self._released.set()
        self._listener = None
        self._stopped = threading.Event()

    def _press(self, k):
        if k != self.key:
            return
        with self._lock:
            if self._held:
                return
            self._held = True
            self._released.clear()
        try:
            self._on_press()
        except Exception:
            log.exception("on_press callback failed")

    def _release(self, k):
        if k != self.key:
            return
        with self._lock:
            if not self._held:
                return
            self._held = False
            self._released.set()
        try:
            self._on_release()
        except Exception:
            log.exception("on_release callback failed")

    def run(self) -> None:
        if self._stopped.is_set():
            return
        with keyboard.Listener(on_press=self._press, on_release=self._release) as listener:
            self._listener = listener
            if self._stopped.is_set():
                listener.stop()
            listener.join()

    def stop(self) -> None:
        self._stopped.set()
        if self._listener is not None:
            self._listener.stop()
        self._released.set()

    def wait_released(self, timeout: float = 0.1) -> bool:
        return self._released.wait(timeout)


def type_text(text: str) -> None:
    """Type text into the currently-focused window via pynput."""
    if not text:
        return
    # Tiny delay so the hotkey's release event has fully propagated before we
    # synthesize keypresses — otherwise the first char can be captured as Alt+X.
    time.sleep(0.05)
    ctrl = keyboard.Controller()
    ctrl.type(text)
