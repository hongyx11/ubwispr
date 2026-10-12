import threading
import unittest
from unittest.mock import Mock

import numpy as np

from ubwispr.app import UbWispr
from ubwispr.config import Config


class AppTests(unittest.TestCase):
    def make_app(self, **kwargs):
        mic = Mock()
        mic.stop.return_value = np.ones(8000, dtype=np.float32)
        asr = Mock()
        asr.transcribe.return_value = 'Hello.'
        statuses = []
        typer = Mock()
        app = UbWispr(Config(), statuses.append, asr=asr, mic=mic, typer=typer, **kwargs)
        app._loaded = True
        self.addCleanup(app.stop)
        return app, mic, asr, typer, statuses

    def test_ignores_repeated_presses_and_duplicate_releases(self):
        app, mic, _, _, _ = self.make_app()
        app._on_press()
        app._on_press()
        app._on_release()
        app._on_release()
        mic.start.assert_called_once()
        mic.stop.assert_called_once()
        self.assertEqual(app._jobs.qsize(), 1)
        app._on_press()  # A pending transcription keeps recording disabled.
        mic.start.assert_called_once()

    def test_limit_stops_and_enqueues_once(self):
        app, mic, _, _, statuses = self.make_app()
        app._on_press()
        app._on_limit()
        app._on_release()
        mic.stop.assert_called_once()
        self.assertEqual(app._jobs.qsize(), 1)
        self.assertTrue(any(s.notify and 'limit' in s.message for s in statuses))

    def test_stale_limit_callback_cannot_stop_a_new_recording(self):
        app, mic, _, _, _ = self.make_app()
        app._on_press()
        previous = app._record_id
        app.set_paused(True)
        app.set_paused(False)
        app._on_press()
        app._on_limit(previous)
        self.assertTrue(app._recording)
        mic.stop.assert_called_once()

    def test_pause_discards_recording(self):
        app, mic, _, _, _ = self.make_app()
        app._on_press()
        app.set_paused(True)
        app._on_release()
        app._on_press()
        mic.stop.assert_called_once()
        mic.start.assert_called_once()
        self.assertTrue(app._jobs.empty())
        app.set_paused(False)
        app._on_press()
        self.assertEqual(mic.start.call_count, 2)

    def test_pause_and_resume_during_inference_discards_stale_text(self):
        app, _, asr, typer, _ = self.make_app()
        def transcribe(*args, **kwargs):
            app.set_paused(True)
            app.set_paused(False)
            return 'Cancelled text'
        asr.transcribe.side_effect = transcribe
        app._handle_audio(np.ones(8000), app._generation)
        typer.assert_not_called()

    def test_short_taps_skip_inference(self):
        app, _, asr, typer, _ = self.make_app()
        app._handle_audio(np.zeros(100), app._generation)
        asr.transcribe.assert_not_called()
        typer.assert_not_called()

    def test_shutdown_suppresses_inflight_transcript(self):
        app, _, asr, typer, _ = self.make_app()
        asr.transcribe.side_effect = lambda *a, **kw: (app.stop(), 'Hello.')[1]
        app._handle_audio(np.ones(8000), app._generation)
        typer.assert_not_called()

    def test_waits_for_hotkey_release_before_typing(self):
        app, _, _, typer, _ = self.make_app()
        app._hotkey = Mock()
        app._hotkey.wait_released.side_effect = [False, False, True]
        app._handle_audio(np.ones(8000), app._generation)
        self.assertEqual(app._hotkey.wait_released.call_count, 3)
        typer.assert_called_once_with('Hello.')

    def test_worker_recovers_after_transcription_failure(self):
        app, _, asr, typer, statuses = self.make_app()
        asr.transcribe.side_effect = [RuntimeError('GPU exhausted'), 'Next recording']
        audio = np.ones(8000)
        app._jobs.put((audio, app._generation))
        thread = threading.Thread(target=app._worker_loop)
        thread.start()
        # Queue the next job once the first has been consumed.
        app._jobs.put((audio, app._generation))
        app._jobs.put(None)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(any(s.state == 'error' and s.notify for s in statuses))
        typer.assert_called_once_with('Next recording')

    def test_stop_cancels_recording_and_timer(self):
        app, mic, _, _, statuses = self.make_app()
        app._on_press()
        app.stop()
        app.stop()
        mic.stop.assert_called_once()
        self.assertIsNone(app._timer)
        self.assertEqual(sum(s.state == 'stopped' for s in statuses), 1)

    def test_model_load_failure_preserves_actionable_status(self):
        app, _, asr, _, statuses = self.make_app()
        asr.load.side_effect = RuntimeError('No model access')
        with self.assertRaisesRegex(RuntimeError, 'No model access'):
            app.run()
        self.assertEqual(statuses[-1].state, 'error')
        self.assertFalse(app.loaded)
