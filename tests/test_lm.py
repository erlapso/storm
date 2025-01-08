import os
import unittest

from knowledge_storm.lm import OpenAIModel
from unittest.mock import MagicMock, patch

class TestOpenAIModel(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.model = "gpt-4"
        self.openai_model = OpenAIModel(model=self.model, api_key=self.api_key)

    @patch('knowledge_storm.lm.OpenAI')
    def test_log_usage(self, mock_openai):
        # Arrange
        mock_response = MagicMock()
        mock_response.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20
        }

        # Act
        self.openai_model.log_usage(mock_response)

        # Assert
        self.assertEqual(self.openai_model.prompt_tokens, 10)
        self.assertEqual(self.openai_model.completion_tokens, 20)

        # Act again to test accumulation
        self.openai_model.log_usage(mock_response)

        # Assert accumulated values
        self.assertEqual(self.openai_model.prompt_tokens, 20)
        self.assertEqual(self.openai_model.completion_tokens, 40)