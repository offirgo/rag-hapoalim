# src/llm/echo.py
from typing import Dict, Any
from .base import BaseLLM


class EchoLLM(BaseLLM):
    """
    Echo LLM for testing - simply returns a summary of its inputs
    """

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Echo prompt back with a simple analysis

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Simple analysis of the prompt
        """
        # Extract the question from the prompt
        question = None
        for line in prompt.split('\n'):
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()

        # Count number of passages
        passage_count = prompt.count("PASSAGE ")

        # Create simple response
        if question:
            return (
                f"[TEST RESPONSE - NO LLM CONNECTED]\n\n"
                f"I received your question: \"{question}\"\n\n"
                f"I found {passage_count} passages in the context. If I were a real LLM, "
                f"I would analyze these passages and provide an answer based on their content."
            )
        else:
            return (
                f"[TEST RESPONSE - NO LLM CONNECTED]\n\n"
                f"I received a prompt with {len(prompt)} characters and {passage_count} passages. "
                f"If I were a real LLM, I would generate a response based on this context."
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "model": "echo-test",
            "type": "test-only",
            "capabilities": "none - just echoes input for testing"
        }