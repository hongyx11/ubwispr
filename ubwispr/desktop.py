"""Qt owns the main thread; dictation runs separately and emits status signals."""
import json
import logging
import os
from pathlib import Path
import signal
import threading

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDialog, QLabel, QMenu, QPushButton, QSystemTrayIcon, QVBoxLayout,
)

from ubwispr.app import UbWispr
from ubwispr.config import Config, config_path
from ubwispr.events import Status
from ubwispr.notifications import notify

log = logging.getLogger(__name__)

_COLORS = {
    "loading": "#97a3b6", "ready": "#a3e6b1", "recording": "#ff8079",
    "transcribing": "#f2cc7e", "paused": "#97a3b6", "error": "#ff8079", "stopped": "#97a3b6",
}


def microphone_icon(state: str) -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#1c2431"))
    painter.drawEllipse(2, 2, 60, 60)
    color = QColor(_COLORS.get(state, _COLORS["ready"]))
    painter.setBrush(color)
    painter.drawRoundedRect(25, 13, 14, 26, 7, 7)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(color, 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawArc(19, 22, 26, 23, 180 * 16, 180 * 16)
    painter.drawLine(32, 45, 32, 51)
    painter.drawLine(25, 51, 39, 51)
    painter.end()
    return QIcon(pixmap)


class Bridge(QObject):
    status = Signal(object)


class StatusWindow(QDialog):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Ubuntu Wispr")
        self.setWindowIcon(microphone_icon("ready"))
        self.setMinimumWidth(380)
        self.setStyleSheet("""
            QDialog { background: #151c27; }
            QLabel { color: #e7edf5; font-size: 14px; }
            QLabel#title { font-size: 27px; font-weight: 600; }
            QLabel#subtitle { color: #a5b2c5; }
            QLabel#status { padding: 16px; background: #222c3b; border-radius: 10px; }
            QPushButton { color: #e7edf5; background: #2a3547; border: none;
                          border-radius: 8px; padding: 11px; font-size: 14px; }
            QPushButton:hover { background: #35445c; }
            QPushButton:disabled { color: #6f7d91; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        title = QLabel("Ubuntu Wispr")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel(f"Hold {controller.config.hotkey_label} · Speak · Release")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        self.status_label = QLabel("Loading speech model…")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)
        self.pause_button = QPushButton("Pause dictation")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(controller.toggle_paused)
        layout.addWidget(self.pause_button)
        configure = QPushButton("Open configuration")
        configure.clicked.connect(controller.open_configuration)
        layout.addWidget(configure)
        quit_button = QPushButton("Quit Ubuntu Wispr")
        quit_button.clicked.connect(QApplication.instance().quit)
        layout.addWidget(quit_button)
        if not controller.has_tray:
            note = QLabel("Tray icons are unavailable. Keep this window open to control dictation.")
            note.setObjectName("subtitle")
            note.setWordWrap(True)
            layout.addWidget(note)

    def closeEvent(self, event):
        if self.controller.has_tray:
            event.ignore()
            self.hide()
        else:
            event.accept()
            QApplication.instance().quit()


class DesktopController(QObject):
    def __init__(self, config: Config, explicit_config: str | None = None):
        super().__init__()
        self.config = config
        self.explicit_config = explicit_config
        self.has_tray = QSystemTrayIcon.isSystemTrayAvailable()
        self.bridge = Bridge(self)
        self.bridge.status.connect(self.update_status)
        self.daemon = UbWispr(config, self.bridge.status.emit)
        self.exit_code = 0
        self.tray = QSystemTrayIcon(microphone_icon("loading"), self)
        self.tray.setToolTip("Ubuntu Wispr — loading speech model")
        self.menu = QMenu()
        self.status_action = self.menu.addAction("Loading speech model…")
        self.status_action.setEnabled(False)
        self.menu.addSeparator()
        self.menu.addAction("Show status", self.show_status)
        self.pause_action = self.menu.addAction("Pause dictation")
        self.pause_action.setCheckable(True)
        self.pause_action.setEnabled(False)
        self.pause_action.triggered.connect(self.daemon.set_paused)
        self.menu.addAction("Open configuration", self.open_configuration)
        self.notification_action = self.menu.addAction("Desktop notifications")
        self.notification_action.setCheckable(True)
        self.notification_action.setChecked(config.notifications)
        self.notification_action.toggled.connect(self.set_notifications)
        self.menu.addSeparator()
        self.menu.addAction("Quit Ubuntu Wispr", QApplication.instance().quit)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self.activate)
        self.window = StatusWindow(self)
        if self.has_tray:
            self.tray.show()
        else:
            log.warning("System tray unavailable; showing the status window")
            self.window.show()
        self.thread = threading.Thread(target=self._run_daemon, daemon=True, name="ubwispr-daemon")

    def _run_daemon(self) -> None:
        try:
            self.daemon.run()
        except Exception:
            self.exit_code = 1

    def start(self) -> None:
        self.thread.start()

    @Slot(object)
    def update_status(self, status: Status) -> None:
        self.tray.setIcon(microphone_icon(status.state))
        self.tray.setToolTip(f"Ubuntu Wispr — {status.message}")
        self.status_action.setText(status.message)
        self.pause_action.setEnabled(self.daemon.loaded and status.state != "stopped")
        self.pause_action.setChecked(self.daemon.paused)
        self.window.status_label.setText(status.message)
        self.window.status_label.setStyleSheet(f"color: {_COLORS.get(status.state, '#e7edf5')};")
        self.window.pause_button.setEnabled(self.pause_action.isEnabled())
        self.window.pause_button.setText("Resume dictation" if self.daemon.paused else "Pause dictation")
        # notify-send uses the standard notification service even on desktops
        # whose tray backend doesn't support QSystemTrayIcon.showMessage().
        notify(status, self.config.notifications)
        if status.state == "error":
            self.show_status()

    def toggle_paused(self) -> None:
        self.daemon.set_paused(not self.daemon.paused)

    def set_notifications(self, enabled: bool) -> None:
        self.config.notifications = enabled

    def show_status(self) -> None:
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def activate(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_status()

    def open_configuration(self) -> None:
        try:
            path = config_path(self.explicit_config)
            if path is None:
                directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "ubwispr"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "config.toml"
                lines = ["# Restart Ubuntu Wispr after editing this file.\n"]
                for name, value in vars(self.config).items():
                    lines.append(f"{name} = {json.dumps(value)}\n")
                # Never overwrite a file created by another process in the meantime.
                try:
                    with path.open("x") as file:
                        file.writelines(lines)
                except FileExistsError:
                    pass
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
                raise RuntimeError(f"Could not open {path}; edit it in your text editor")
        except (OSError, ValueError, RuntimeError) as exc:
            self.update_status(Status("error", str(exc), notify=True))

    def stop(self) -> None:
        self.daemon.stop()
        self.tray.hide()


def run_desktop(config: Config, explicit_config: str | None = None) -> int:
    app = QApplication([])
    app.setApplicationName("Ubuntu Wispr")
    app.setQuitOnLastWindowClosed(False)
    controller = DesktopController(config, explicit_config)
    app.aboutToQuit.connect(controller.stop)
    # Allow Python signal handling while Qt is waiting in its native event loop.
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(200)
    previous = {sig: signal.signal(sig, lambda *_: app.quit()) for sig in (signal.SIGINT, signal.SIGTERM)}
    QTimer.singleShot(0, controller.start)
    try:
        result = app.exec()
        return result or controller.exit_code
    finally:
        controller.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
