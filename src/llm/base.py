# src/llm/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseLLM(ABC):
    """
    Base abstract class for LLM implementations
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters for generation

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        pass


class LLMError(Exception):
    """Exception raised for LLM-related errors"""
    pass