import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication

from ubwispr.app import UbWispr
from ubwispr.config import Config, load
from ubwispr.desktop import DesktopController
from ubwispr.events import Status


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.qt.setQuitOnLastWindowClosed(False)

    def setUp(self):
        def factory(config, callback):
            return UbWispr(config, callback, asr=Mock(), mic=Mock(), typer=Mock())
        self.notify_patch = patch('ubwispr.desktop.notify')
        self.notify = self.notify_patch.start()
        self.addCleanup(self.notify_patch.stop)
        with patch('ubwispr.desktop.UbWispr', side_effect=factory), \
             patch('ubwispr.desktop.QSystemTrayIcon.isSystemTrayAvailable', return_value=False):
            self.controller = DesktopController(Config())
        self.addCleanup(self.cleanup_controller)

    def cleanup_controller(self):
        self.controller.stop()
        self.controller.window.hide()
        self.controller.window.deleteLater()
        self.controller.deleteLater()
        self.qt.processEvents()

    def test_fallback_window_shows_status_and_controls_pause(self):
        self.assertTrue(self.controller.window.isVisible())
        self.controller.daemon._loaded = True
        self.controller.update_status(Status('ready', 'Ready'))
        self.assertTrue(self.controller.pause_action.isEnabled())
        self.controller.window.pause_button.click()
        self.qt.processEvents()
        self.assertTrue(self.controller.daemon.paused)
        self.assertEqual(self.controller.window.pause_button.text(), 'Resume dictation')
        self.controller.window.pause_button.click()
        self.qt.processEvents()
        self.assertFalse(self.controller.daemon.paused)

    def test_background_status_signal_updates_widgets(self):
        thread = threading.Thread(target=lambda: self.controller.bridge.status.emit(Status('recording', 'Recording')))
        thread.start()
        thread.join(1)
        self.qt.processEvents()
        self.assertEqual(self.controller.window.status_label.text(), 'Recording')
        self.assertFalse(self.controller.tray.icon().isNull())

    def test_configuration_button_creates_valid_toml_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {'XDG_CONFIG_HOME': tmp, 'UBWISPR_CONFIG': ''}), \
             patch('ubwispr.desktop.config_path', return_value=None), \
             patch('ubwispr.desktop.QDesktopServices.openUrl', return_value=True):
            self.controller.open_configuration()
            path = Path(tmp) / 'ubwispr' / 'config.toml'
            self.assertEqual(load(str(path)), self.controller.config)
            path.write_text('hotkey = "f9"\n')
            self.controller.open_configuration()
            self.assertEqual(path.read_text(), 'hotkey = "f9"\n')

    def test_error_status_opens_window_and_requests_notification(self):
        self.controller.window.hide()
        status = Status('error', 'Microphone unavailable', notify=True)
        self.controller.update_status(status)
        self.assertTrue(self.controller.window.isVisible())
        self.assertEqual(self.controller.window.status_label.text(), status.message)
        self.notify.assert_called_with(status, True)
