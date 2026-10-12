import unittest
from unittest.mock import patch

import httpx
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from ubwispr.model_access import ModelAccessError, access_error, check_model_access

MODEL = 'CohereLabs/cohere-transcribe-03-2026'


def gated_error(code=401):
    response = httpx.Response(code, request=httpx.Request('HEAD', f'https://huggingface.co/{MODEL}/resolve/main/config.json'))
    return GatedRepoError('Raw response and request identifiers', response=response)


class ModelAccessTests(unittest.TestCase):
    def test_missing_token_explains_both_required_setup_steps(self):
        with patch('ubwispr.model_access.get_token', return_value=None):
            message = str(access_error(MODEL, gated_error()))
        self.assertIn('No Hugging Face token found', message)
        self.assertIn('accept/request model access', message)
        self.assertIn('hf auth login', message)
        self.assertNotIn('Raw response', message)

    def test_existing_token_explains_account_access_and_permissions(self):
        with patch('ubwispr.model_access.get_token', return_value='test-placeholder'):
            message = str(access_error(MODEL, gated_error(403)))
        self.assertIn('account associated with your token', message)
        self.assertIn('read access', message)
        self.assertNotIn('test-placeholder', message)

    def test_finds_hub_error_wrapped_by_transformers(self):
        wrapped = OSError('Model could not load')
        wrapped.__cause__ = gated_error()
        with patch('ubwispr.model_access.get_token', return_value=None):
            self.assertIsInstance(access_error(MODEL, wrapped), ModelAccessError)

    def test_unrelated_load_errors_remain_unchanged(self):
        self.assertIsNone(access_error(MODEL, RuntimeError('CUDA out of memory')))

    def test_probe_reports_gated_access_without_downloading(self):
        with patch('ubwispr.model_access.get_hf_file_metadata', side_effect=gated_error()) as metadata, \
             patch('ubwispr.model_access.get_token', return_value=None):
            ok, message = check_model_access(MODEL)
        self.assertFalse(ok)
        self.assertIn('hf auth login', message)
        self.assertEqual(metadata.call_args.kwargs['timeout'], 5)

    def test_probe_does_not_call_network_failures_authentication_errors(self):
        with patch('ubwispr.model_access.get_hf_file_metadata', side_effect=TimeoutError()):
            ok, message = check_model_access(MODEL)
        self.assertFalse(ok)
        self.assertIn('network', message)
        self.assertNotIn('auth login', message)

    def test_asr_load_converts_wrapped_gated_error(self):
        from ubwispr.asr import AsrEngine
        wrapped = OSError('Access to gated repo')
        wrapped.__cause__ = gated_error()
        with patch('ubwispr.asr.AutoProcessor.from_pretrained', side_effect=wrapped), \
             patch('ubwispr.model_access.get_token', return_value=None):
            with self.assertRaisesRegex(ModelAccessError, 'hf auth login'):
                AsrEngine(MODEL).load()

    def test_asr_keeps_unrelated_load_failure(self):
        from ubwispr.asr import AsrEngine
        error = RuntimeError('CUDA out of memory')
        with patch('ubwispr.asr.AutoProcessor.from_pretrained', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                AsrEngine(MODEL).load()
        self.assertIs(caught.exception, error)
