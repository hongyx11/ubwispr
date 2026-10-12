import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ubwispr.config import Config, load


class ConfigTests(unittest.TestCase):
    def test_rejects_invalid_values(self):
        for field, value in (
            ('sample_rate', True), ('max_new_tokens', 0), ('max_record_seconds', float('inf')),
            ('min_record_seconds', -1), ('dtype', 'bf16'), ('device', 'gpu'),
            ('hotkey', 'fn'), ('notifications', 'yes'), ('model_id', ''),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                Config(**{field: value}).validate()

    def test_rejects_inverted_durations(self):
        with self.assertRaises(ValueError):
            Config(min_record_seconds=10, max_record_seconds=5).validate()

    def test_explicit_config_overrides_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / 'selected.toml'
            selected.write_text('hotkey = "f9"\nnotifications = false\n')
            with patch.dict(os.environ, {'UBWISPR_CONFIG': '/missing/config.toml'}):
                config = load(str(selected))
            self.assertEqual(config.hotkey, 'f9')
            self.assertFalse(config.notifications)

    def test_explicit_missing_config_does_not_silently_fall_back(self):
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            load('/missing/ubwispr.toml')

    def test_unknown_fields_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.toml'
            path.write_text('langauge = "en"\n')
            with self.assertRaisesRegex(ValueError, 'langauge'):
                load(str(path))

    def test_uses_xdg_config_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ubwispr' / 'config.toml'
            path.parent.mkdir()
            path.write_text('language = "fr"\n')
            with patch.dict(os.environ, {'XDG_CONFIG_HOME': tmp, 'UBWISPR_CONFIG': ''}):
                self.assertEqual(load().language, 'fr')
