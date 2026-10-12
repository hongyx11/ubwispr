import argparse
import logging
import signal

from ubwispr.config import load
from ubwispr.model_access import check_model_access
from ubwispr.runtime import InstanceLock, check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ubuntu Wispr — hold-to-dictate for X11")
    parser.add_argument("--check", action="store_true", help="Check setup without loading or downloading the model")
    parser.add_argument("--config", metavar="PATH", help="Use a specific TOML configuration")
    parser.add_argument("--no-tray", action="store_true", help="Run without the tray interface")
    parser.add_argument("--no-notifications", action="store_true", help="Disable desktop notifications")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("transformers", "httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        cfg = load(args.config)
        if args.no_notifications:
            cfg.notifications = False
        checks = check(cfg)
        if args.check:
            checks.append(check_model_access(cfg.model_id))
            for ok, message in checks:
                print(f"{'OK' if ok else 'FAIL'}  {message}")
            return 0 if all(ok for ok, _ in checks) else 1
        problems = [message for ok, message in checks if not ok]
        if problems:
            raise RuntimeError("; ".join(problems))
        with InstanceLock():
            if not args.no_tray:
                try:
                    from ubwispr.desktop import run_desktop
                except ModuleNotFoundError as exc:
                    if exc.name != "PySide6":
                        raise
                    logging.getLogger("ubwispr").warning(
                        "Tray unavailable; in the ubwispr Conda environment run python -m pip install -e '.[desktop]'"
                    )
                else:
                    return run_desktop(cfg, args.config)
            from ubwispr.app import UbWispr
            from ubwispr.notifications import notify
            daemon = UbWispr(cfg, lambda status: notify(status, cfg.notifications))
            previous = signal.signal(signal.SIGTERM, lambda *_: daemon.stop())
            try:
                daemon.run()
            finally:
                signal.signal(signal.SIGTERM, previous)
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logging.getLogger("ubwispr").error("%s", exc)
        if not args.check:
            from ubwispr.events import Status
            from ubwispr.notifications import notify
            enabled = not args.no_notifications and (cfg.notifications if 'cfg' in locals() else True)
            notify(Status("error", str(exc), notify=True), enabled)
        return 1
