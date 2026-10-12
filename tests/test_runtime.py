import os
import tempfile
import unittest
from unittest.mock import patch

from ubwispr.runtime import InstanceLock


class InstanceTests(unittest.TestCase):
    def test_second_instance_is_rejected_and_lock_is_reusable(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'XDG_CACHE_HOME': tmp}):
            with InstanceLock():
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    with InstanceLock():
                        pass
            with InstanceLock():
                pass
