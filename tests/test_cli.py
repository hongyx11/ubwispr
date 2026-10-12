import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from ubwispr.cli import main
from ubwispr.config import Config


class CliTests(unittest.TestCase):
    def test_check_reports_failure_without_initializing_the_model(self):
        output = io.StringIO()
        with patch('ubwispr.cli.load', return_value=Config()), \
             patch('ubwispr.cli.check', return_value=[(True, 'CUDA available'), (False, 'No X11')]), \
             patch('ubwispr.cli.check_model_access', return_value=(True, 'Model accessible')), \
             patch('ubwispr.app.UbWispr') as daemon, contextlib.redirect_stdout(output):
            self.assertEqual(main(['--check']), 1)
            daemon.assert_not_called()
        self.assertIn('OK  CUDA available', output.getvalue())
        self.assertIn('FAIL  No X11', output.getvalue())

    def test_check_reports_missing_model_access(self):
        output = io.StringIO()
        with patch('ubwispr.cli.load', return_value=Config()), \
             patch('ubwispr.cli.check', return_value=[(True, 'CUDA available')]), \
             patch('ubwispr.cli.check_model_access', return_value=(False, 'Run hf auth login')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(['--check']), 1)
        self.assertIn('FAIL  Run hf auth login', output.getvalue())

    def test_startup_error_requests_notification(self):
        with patch('ubwispr.cli.load', return_value=Config()), \
             patch('ubwispr.cli.check', return_value=[(False, 'No X11')]), \
             patch('ubwispr.notifications.notify') as notify:
            self.assertEqual(main([]), 1)
            self.assertEqual(notify.call_args.args[0].message, 'No X11')

    def test_disabled_notifications_are_respected_for_errors(self):
        with patch('ubwispr.cli.load', side_effect=ValueError('Invalid TOML')), \
             patch('ubwispr.notifications.notify') as notify:
            self.assertEqual(main(['--no-notifications']), 1)
            self.assertFalse(notify.call_args.args[1])
