import logging
import shutil
import subprocess

from ubwispr.events import Status

log = logging.getLogger(__name__)


def notify(status: Status, enabled: bool = True) -> None:
    """Use the desktop notification service, including when the tray is absent."""
    executable = shutil.which("notify-send")
    if not enabled or not status.notify or not executable:
        return
    try:
        subprocess.run(
            [executable, "--app-name=Ubuntu Wispr", "--icon=audio-input-microphone",
             "--expire-time=4000", "--", "Ubuntu Wispr", status.message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        log.debug("Desktop notification unavailable")
