import unittest

from knowledge_storm.lm import OpenAIModel
from unittest.mock import MagicMock, patch

class TestOpenAIModel(unittest.TestCase):
    @patch('knowledge_storm.lm.OpenAI')
    def test_token_usage_tracking(self, mock_openai):
        # Arrange
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = {
            'choices': [{'message': {'content': 'Test response'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}
        }
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIModel(model="gpt-3.5-turbo", api_key="test_key")

        # Act
        result = model("Test prompt")

        # Assert
        self.assertEqual(result, ['Test response'])
        self.assertEqual(model.prompt_tokens, 10)
        self.assertEqual(model.completion_tokens, 5)

        # Act again to check accumulation
        result = model("Another test prompt")

        # Assert
        self.assertEqual(result, ['Test response'])
        self.assertEqual(model.prompt_tokens, 20)
        self.assertEqual(model.completion_tokens, 10)

        # Check usage reset
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {'gpt-3.5-turbo': {'prompt_tokens': 20, 'completion_tokens': 10}})
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)