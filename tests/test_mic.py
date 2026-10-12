import io
import subprocess
import threading
import unittest
from unittest.mock import patch

import numpy as np

from ubwispr.mic import MicRecorder


class FakeProcess:
    def __init__(self, payload=b'', exit_code=0, fail_to_stop=False):
        self.stdout = io.BytesIO(payload)
        self.returncode = None
        self.exit_code = exit_code
        self.fail_to_stop = fail_to_stop
        self.terminated = threading.Event()
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated.set()

    def wait(self, timeout=None):
        if self.fail_to_stop and not self.killed:
            raise subprocess.TimeoutExpired('arecord', timeout)
        self.returncode = self.exit_code
        return self.returncode

    def kill(self):
        self.killed = True
        self.terminated.set()


class FinalBlock(io.BytesIO):
    def __init__(self, payload, terminated):
        super().__init__(payload)
        self.terminated = terminated

    def read1(self, size):
        if not self.terminated.wait(2):
            raise TimeoutError('Recording was never terminated')
        return super().read1(size)


class MicTests(unittest.TestCase):
    def test_release_drains_the_final_short_audio_block(self):
        samples = np.array([0, 32767, -32768, 1234], dtype='<i2')
        proc = FakeProcess()
        proc.stdout = FinalBlock(samples.tobytes(), proc.terminated)
        with patch('ubwispr.mic.subprocess.Popen', return_value=proc):
            mic = MicRecorder()
            mic.start()
            audio = mic.stop()
        np.testing.assert_array_equal(audio, samples.astype(np.float32) / 32768)
        self.assertTrue(proc.stdout.closed)

    def test_capture_is_bounded_and_odd_final_byte_is_ignored(self):
        proc = FakeProcess(b'\x01\x00' * 100 + b'\x01')
        with patch('ubwispr.mic.subprocess.Popen', return_value=proc) as spawn:
            mic = MicRecorder(sample_rate=10, max_seconds=0.55)
            mic.start()
            audio = mic.stop()
        self.assertEqual(len(audio), 5)
        self.assertIn('-d', spawn.call_args.args[0])
        self.assertEqual(spawn.call_args.args[0][-1], '1')

    def test_reports_alsa_failure_and_allows_retry(self):
        proc = FakeProcess(exit_code=1)
        def spawn(*args, **kwargs):
            kwargs['stderr'].write(b'Unknown PCM missing-device')
            return proc
        with patch('ubwispr.mic.subprocess.Popen', side_effect=spawn):
            mic = MicRecorder()
            mic.start()
            with self.assertRaisesRegex(RuntimeError, 'Unknown PCM'):
                mic.stop()
        self.assertIsNone(mic._proc)
        self.assertEqual(mic.stop().size, 0)

    def test_force_kills_a_stuck_recorder(self):
        proc = FakeProcess(fail_to_stop=True)
        with patch('ubwispr.mic.subprocess.Popen', return_value=proc):
            mic = MicRecorder()
            mic.start()
            mic.stop()
        self.assertTrue(proc.killed)

    def test_release_accepts_arecord_exit_one_with_valid_audio_and_no_errors(self):
        samples = np.array([100, -200, 300], dtype='<i2')
        proc = FakeProcess(exit_code=1)
        proc.stdout = FinalBlock(samples.tobytes(), proc.terminated)
        with patch('ubwispr.mic.subprocess.Popen', return_value=proc):
            mic = MicRecorder()
            mic.start()
            audio = mic.stop()
        np.testing.assert_array_equal(audio, samples.astype(np.float32) / 32768)

    def test_fast_tap_can_stop_before_any_audio_arrives(self):
        proc = FakeProcess(exit_code=1)
        with patch('ubwispr.mic.subprocess.Popen', return_value=proc):
            mic = MicRecorder()
            mic.start()
            self.assertEqual(mic.stop().size, 0)

    def test_release_accepts_interrupted_pcm_read(self):
        proc = FakeProcess(b'\x01\x00' * 20, exit_code=1)
        def spawn(*args, **kwargs):
            self.assertEqual(kwargs['env']['LC_ALL'], 'C')
            kwargs['stderr'].write(b'arecord: pcm_read:2240: read error: Interrupted system call\n')
            return proc
        with patch('ubwispr.mic.subprocess.Popen', side_effect=spawn):
            mic = MicRecorder()
            mic.start()
            self.assertEqual(mic.stop().size, 20)

    def test_interruption_does_not_mask_other_alsa_errors(self):
        proc = FakeProcess(b'\x01\x00' * 20, exit_code=1)
        def spawn(*args, **kwargs):
            kwargs['stderr'].write(
                b'arecord: pcm_read:2240: read error: Interrupted system call\n'
                b'arecord: pcm_read:2240: read error: Input/output error\n'
            )
            return proc
        with patch('ubwispr.mic.subprocess.Popen', side_effect=spawn):
            mic = MicRecorder()
            mic.start()
            with self.assertRaisesRegex(RuntimeError, 'Input/output error'):
                mic.stop()

    def test_spontaneous_exit_one_is_still_a_failure(self):
        proc = FakeProcess(b'\x01\x00' * 20, exit_code=1)
        proc.returncode = 1
        with patch('ubwispr.mic.subprocess.Popen', return_value=proc):
            mic = MicRecorder()
            mic.start()
            with self.assertRaisesRegex(RuntimeError, 'Microphone capture failed'):
                mic.stop()
        self.assertFalse(proc.terminated.is_set())

    def test_alsa_errors_are_reported_even_after_some_audio_arrives(self):
        proc = FakeProcess(b'\x01\x00' * 20, exit_code=1)
        def spawn(*args, **kwargs):
            kwargs['stderr'].write(b'arecord: pcm_read: Input/output error')
            return proc
        with patch('ubwispr.mic.subprocess.Popen', side_effect=spawn):
            mic = MicRecorder()
            mic.start()
            with self.assertRaisesRegex(RuntimeError, 'Input/output error'):
                mic.stop()
