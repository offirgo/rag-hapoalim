# src/llm/ollama.py
import json
import logging
import requests
from typing import Dict, Any, Optional, List

from .base import BaseLLM, LLMError

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """
    Ollama-based LLM implementation

    Requires Ollama to be installed and running locally:
    https://github.com/ollama/ollama
    """

    def __init__(
            self,
            model: str = "tinyllama",
            base_url: str = "http://localhost:11434",
            temperature: float = 0.7,
            max_tokens: int = 1024
    ):
        """
        Initialize Ollama LLM

        Args:
            model: Ollama model to use (e.g., "llama3", "mistral")
            base_url: Ollama API URL
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Verify Ollama is available
        self._check_availability()

        logger.info(f"OllamaLLM initialized with model '{model}'")

    def _check_availability(self) -> None:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise LLMError(f"Ollama server returned status {response.status_code}")

            # Check if our model is available
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]

            if self.model not in available_models:
                logger.warning(
                    f"Model '{self.model}' not found in available models: {available_models}. "
                    f"You may need to run: ollama pull {self.model}"
                )

        except requests.RequestException as e:
            raise LLMError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Please install Ollama (https://ollama.ai) and start it with 'ollama serve'"
            ) from e

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt using Ollama

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters for generation

        Returns:
            Generated text response
        """
        # Merge default parameters with any overrides
        params = {
            "model": self.model,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=params,
                timeout=60  # 60 second timeout
            )

            if response.status_code != 200:
                raise LLMError(f"Ollama API error: {response.status_code} - {response.text}")

            result = response.json()
            return result.get("response", "")

        except requests.RequestException as e:
            raise LLMError(f"Ollama API request failed: {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.get(f"{self.base_url}/api/show", params={"name": self.model})
            if response.status_code != 200:
                return {"model": self.model, "error": f"Status {response.status_code}"}

            info = response.json()
            return {
                "model": self.model,
                "parameter_size": info.get("parameters"),
                "quantization": info.get("quantization", "unknown"),
                "license": info.get("license", "unknown")
            }

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"model": self.model, "error": str(e)}