import unittest
from unittest.mock import patch, MagicMock

from requests.exceptions import RequestException

from src.llm.ollama_llm import OllamaLLM, LLMError


class TestOllamaLLM(unittest.TestCase):
    @patch('requests.get')
    def test_check_availability_success(self, mock_get):
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "tinyllama"}]}
        mock_get.return_value = mock_response

        # Should not raise an exception
        llm = OllamaLLM(model="tinyllama")
        self.assertEqual(llm.model, "tinyllama")

    @patch('requests.get')
    def test_check_availability_model_not_found(self, mock_get):
        # Mock a response where model isn't available
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "mistral"}]}
        mock_get.return_value = mock_response

        # Should warn but not raise an exception
        with self.assertLogs(level='WARNING') as cm:
            llm = OllamaLLM(model="tinyllama")
            self.assertIn("not found in available models", cm.output[0])

    @patch('requests.get')
    def test_check_availability_error(self, mock_get):
        # Mock a failed response
        mock_get.side_effect = RequestException("Connection error")

        # Should raise LLMError
        with self.assertRaises(LLMError):
            OllamaLLM(model="llama3")

    @patch('requests.post')
    def test_generate(self, mock_post):
        # Mock a successful generation
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Generated text"}
        mock_post.return_value = mock_response

        # Patch the _check_availability method
        with patch.object(OllamaLLM, '_check_availability'):
            llm = OllamaLLM()
            result = llm.generate("Test prompt")
            self.assertEqual(result, "Generated text")

            # Check the request parameters
            args, kwargs = mock_post.call_args
            self.assertEqual(kwargs['json']['prompt'], "Test prompt")

    @patch('requests.post')
    def test_generate_error(self, mock_post):
        # Mock a failed generation
        mock_post.side_effect = RequestException("API error")

        # Patch the _check_availability method
        with patch.object(OllamaLLM, '_check_availability'):
            llm = OllamaLLM()
            with self.assertRaises(LLMError):
                llm.generate("Test prompt")

    @patch('requests.get')
    def test_get_model_info(self, mock_get):
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "parameters": "7B",
            "quantization": "Q4_0",
            "license": "Apache 2.0"
        }
        mock_get.return_value = mock_response

        # Patch the _check_availability method
        with patch.object(OllamaLLM, '_check_availability'):
            llm = OllamaLLM()
            info = llm.get_model_info()
            self.assertEqual(info["model"], "tinyllama")
            self.assertEqual(info["parameter_size"], "7B")