import unittest

from knowledge_storm.lm import OpenAIModel
from unittest.mock import MagicMock, patch

class TestOpenAIModel(unittest.TestCase):
    @patch('knowledge_storm.lm.OpenAI')
    def test_token_usage_tracking(self, mock_openai):
        # Create a mock response with usage data
        mock_response = MagicMock()
        mock_response.get.return_value = {
            'prompt_tokens': 10,
            'completion_tokens': 20
        }

        # Configure the mock OpenAI client
        mock_client = MagicMock()
        mock_client.request.return_value = {'choices': [{'text': 'Test completion'}], 'usage': mock_response.get()}
        mock_openai.return_value = mock_client

        # Create an instance of OpenAIModel
        model = OpenAIModel(model="gpt-3.5-turbo")

        # Call the model
        result = model("Test prompt")

        # Assert the result
        self.assertEqual(result, ['Test completion'])

        # Check if token usage is tracked correctly
        self.assertEqual(model.prompt_tokens, 10)
        self.assertEqual(model.completion_tokens, 20)

        # Check if get_usage_and_reset works correctly
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {'gpt-3.5-turbo': {'prompt_tokens': 10, 'completion_tokens': 20}})

        # Ensure usage is reset after calling get_usage_and_reset
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)