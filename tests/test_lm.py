import os
import threading
import unittest

from concurrent.futures import ThreadPoolExecutor
from knowledge_storm.lm import AzureOpenAIModel, ClaudeModel, DeepSeekModel, GoogleModel, GroqModel, OllamaClient, OpenAIModel, TogetherClient, VLLMClient
from unittest.mock import MagicMock, patch

class TestOpenAIModel(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_api_key"})
    @patch('openai.OpenAI')
    def test_token_usage_tracking(self, mock_openai):
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20
        }
        mock_response.choices = [MagicMock(text="Test response")]
        mock_openai.return_value.completions.create.return_value = mock_response

        # Create an instance of OpenAIModel
        model = OpenAIModel(model="gpt-3.5-turbo")

        # Make a call to the model
        result = model("Test prompt")

        # Check if the result is correct
        self.assertEqual(result, ["Test response"])

        # Check if the token usage is correctly logged
        self.assertEqual(model.prompt_tokens, 10)
        self.assertEqual(model.completion_tokens, 20)

        # Get usage and check if it's correct
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {
            "gpt-3.5-turbo": {
                "prompt_tokens": 10,
                "completion_tokens": 20
            }
        })

        # Check if the token usage is reset
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_api_key"})
    @patch('openai.OpenAI')
    def test_concurrent_token_usage_tracking(self, mock_openai):
        # Mock the OpenAI API response
        def mock_completion(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 20
            }
            mock_response.choices = [MagicMock(text="Test response")]
            return mock_response

        mock_openai.return_value.completions.create.side_effect = mock_completion

        # Create an instance of OpenAIModel
        model = OpenAIModel(model="gpt-3.5-turbo")

        # Function to make API calls
        def make_api_call():
            model("Test prompt")

        # Make multiple concurrent API calls
        num_calls = 5
        with ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = [executor.submit(make_api_call) for _ in range(num_calls)]
            # Wait for all calls to complete
            for future in futures:
                future.result()

        # Check if the token usage is correctly accumulated
        self.assertEqual(model.prompt_tokens, 10 * num_calls)
        self.assertEqual(model.completion_tokens, 20 * num_calls)

        # Get usage and check if it's correct
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {
            "gpt-3.5-turbo": {
                "prompt_tokens": 10 * num_calls,
                "completion_tokens": 20 * num_calls
            }
        })

        # Check if the token usage is reset
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "fake_azure_api_key",
        "AZURE_OPENAI_API_BASE": "https://fake-azure-endpoint.openai.azure.com/",
        "AZURE_OPENAI_API_VERSION": "2023-05-15"
    })
    @patch('openai.AzureOpenAI')
    def test_azure_openai_token_usage_tracking(self, mock_azure_openai):
        # Mock the Azure OpenAI API response
        mock_response = MagicMock()
        mock_response.usage = {
            "prompt_tokens": 15,
            "completion_tokens": 25
        }
        mock_response.choices = [MagicMock(message=MagicMock(content="Azure OpenAI response"))]
        mock_azure_openai.return_value.chat.completions.create.return_value = mock_response

        # Create an instance of AzureOpenAIModel
        model = AzureOpenAIModel(model="gpt-4", api_version="2023-05-15")

        # Make a call to the model
        result = model("Test prompt for Azure OpenAI")

        # Check if the result is correct
        self.assertEqual(result, ["Azure OpenAI response"])

        # Check if the token usage is correctly logged
        self.assertEqual(model.prompt_tokens, 15)
        self.assertEqual(model.completion_tokens, 25)

        # Get usage and check if it's correct
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {
            "gpt-4": {
                "prompt_tokens": 15,
                "completion_tokens": 25
            }
        })

        # Check if the token usage is reset
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)

class TestDeepSeekModel(unittest.TestCase):
    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake_deepseek_api_key"})
    @patch('requests.post')
    def test_deepseek_model(self, mock_post):
        # Mock the DeepSeek API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "DeepSeek response"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 18
            }
        }
        mock_post.return_value = mock_response

        # Create an instance of DeepSeekModel
        model = DeepSeekModel(model="deepseek-chat")

        # Make a call to the model
        result = model("Test prompt for DeepSeek")

        # Check if the result is correct
        self.assertEqual(result, ["DeepSeek response"])

        # Check if the token usage is correctly logged
        self.assertEqual(model.prompt_tokens, 12)
        self.assertEqual(model.completion_tokens, 18)

        # Get usage and check if it's correct
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {
            "deepseek-chat": {
                "prompt_tokens": 12,
                "completion_tokens": 18
            }
        })

        # Check if the token usage is reset
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)

        # Verify that the API was called with the correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(call_args[1]['headers']['Authorization'], "Bearer fake_deepseek_api_key")
        self.assertEqual(call_args[1]['json']['messages'][0]['content'], "Test prompt for DeepSeek")

class TestGroqModel(unittest.TestCase):
    @patch.dict('os.environ', {"GROQ_API_KEY": "fake_groq_api_key"})
    @patch('requests.post')
    def test_groq_model_token_usage_and_api_call(self, mock_post):
        # Mock the Groq API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Groq response"}}],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25
            }
        }
        mock_post.return_value = mock_response

        # Create an instance of GroqModel
        model = GroqModel(model="llama3-70b-8192")

        # Make a call to the model
        result = model("Test prompt for Groq")

        # Check if the result is correct
        self.assertEqual(result, ["Groq response"])

        # Check if the token usage is correctly logged
        self.assertEqual(model.prompt_tokens, 15)
        self.assertEqual(model.completion_tokens, 25)

        # Get usage and check if it's correct
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {
            "llama3-70b-8192": {
                "prompt_tokens": 15,
                "completion_tokens": 25
            }
        })

        # Check if the token usage is reset
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)

        # Verify that the API was called with the correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(call_args[1]['headers']['Authorization'], "Bearer fake_groq_api_key")
        self.assertEqual(call_args[1]['json']['messages'][0]['content'], "Test prompt for Groq")
        self.assertEqual(call_args[1]['json']['model'], "llama3-70b-8192")

class TestVLLMClient(unittest.TestCase):
    @patch('knowledge_storm.lm.OpenAI')
    def test_vllm_client_initialization_and_call(self, mock_openai):
        # Mock the OpenAI client's chat.completions.create method
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="VLLM response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_openai.return_value.chat.completions.create.return_value = mock_completion

        # Create an instance of VLLMClient
        client = VLLMClient(model="test-model", port=8000, url="http://localhost")

        # Check if the client is initialized correctly
        self.assertEqual(client.model, "test-model")
        self.assertEqual(client.base_url, "http://localhost:8000/v1/chat/")

        # Make a call to the client
        result = client("Test prompt for VLLM")

        # Check if the result is correct
        self.assertEqual(result, ["VLLM response"])

        # Check if the token usage is correctly logged
        self.assertEqual(client.prompt_tokens, 10)
        self.assertEqual(client.completion_tokens, 20)

        # Get usage and check if it's correct
        usage = client.get_usage_and_reset()
        self.assertEqual(usage, {
            "test-model": {
                "prompt_tokens": 10,
                "completion_tokens": 20
            }
        })

        # Check if the token usage is reset
        self.assertEqual(client.prompt_tokens, 0)
        self.assertEqual(client.completion_tokens, 0)

        # Verify that the OpenAI client was called with the correct parameters
        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt for VLLM"}]
        )

class TestClaudeModel(unittest.TestCase):
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake_anthropic_api_key"})
    @patch('anthropic.Anthropic')
    def test_claude_model_token_usage_and_api_call(self, mock_anthropic):
        # Mock the Anthropic API response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Claude response")]
        mock_response.usage.input_tokens = 18
        mock_response.usage.output_tokens = 22
        mock_anthropic.return_value.messages.create.return_value = mock_response

        # Create an instance of ClaudeModel
        model = ClaudeModel(model="claude-2")

        # Make a call to the model
        result = model("Test prompt for Claude")

        # Check if the result is correct
        self.assertEqual(result, ["Claude response"])

        # Check if the token usage is correctly logged
        self.assertEqual(model.prompt_tokens, 18)
        self.assertEqual(model.completion_tokens, 22)

        # Get usage and check if it's correct
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {
            "claude-2": {
                "prompt_tokens": 18,
                "completion_tokens": 22
            }
        })

        # Check if the token usage is reset
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)

        # Verify that the API was called with the correct parameters
        mock_anthropic.return_value.messages.create.assert_called_once()
        call_kwargs = mock_anthropic.return_value.messages.create.call_args[1]
        self.assertEqual(call_kwargs['messages'][0]['content'], "Test prompt for Claude")
        self.assertEqual(call_kwargs['model'], "claude-2")

class TestTogetherClient(unittest.TestCase):
    @patch.dict(os.environ, {"TOGETHER_API_KEY": "fake_together_api_key"})
    @patch('google.generativeai', new_callable=MagicMock)
    def test_together_client_initialization_and_call(self, mock_genai):
        # Mock the GenerativeModel's generate_content method
        mock_response = MagicMock()
        mock_response.parts = [MagicMock(text="Together AI response")]
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 25
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        # Create an instance of TogetherClient
        client = TogetherClient(model="together-ai-model")

        # Check if the client is initialized correctly
        self.assertEqual(client.model, "together-ai-model")

        # Make a call to the client
        result = client("Test prompt for Together AI")

        # Check if the result is correct
        self.assertEqual(result, ["Together AI response"])

        # Check if the token usage is correctly logged
        self.assertEqual(client.prompt_tokens, 15)
        self.assertEqual(client.completion_tokens, 25)

        # Get usage and check if it's correct
        usage = client.get_usage_and_reset()
        self.assertEqual(usage, {
            "together-ai-model": {
                "prompt_tokens": 15,
                "completion_tokens": 25
            }
        })

        # Check if the token usage is reset
        self.assertEqual(client.prompt_tokens, 0)
        self.assertEqual(client.completion_tokens, 0)

        # Verify that the GenerativeModel was called with the correct parameters
        mock_genai.GenerativeModel.return_value.generate_content.assert_called_once_with(
            "Test prompt for Together AI",
            generation_config=mock_genai.GenerationConfig.return_value
        )

class TestGoogleModel(unittest.TestCase):
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_api_key"})
    @patch('google.generativeai')
    def test_google_model_initialization_and_call(self, mock_genai):
        # Mock the GenerativeModel's generate_content method
        mock_response = MagicMock()
        mock_response.parts = [MagicMock(text="Google Gemini response")]
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 30
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        # Create an instance of GoogleModel
        model = GoogleModel(model="gemini-pro")

        # Check if the model is initialized correctly
        self.assertEqual(model.model, "gemini-pro")

        # Make a call to the model
        result = model("Test prompt for Google Gemini")

        # Check if the result is correct
        self.assertEqual(result, ["Google Gemini response"])

        # Check if the token usage is correctly logged
        self.assertEqual(model.prompt_tokens, 20)
        self.assertEqual(model.completion_tokens, 30)

        # Get usage and check if it's correct
        usage = model.get_usage_and_reset()
        self.assertEqual(usage, {
            "gemini-pro": {
                "prompt_tokens": 20,
                "completion_tokens": 30
            }
        })

        # Check if the token usage is reset
        self.assertEqual(model.prompt_tokens, 0)
        self.assertEqual(model.completion_tokens, 0)

        # Verify that the GenerativeModel was called with the correct parameters
        mock_genai.GenerativeModel.return_value.generate_content.assert_called_once_with(
            "Test prompt for Google Gemini",
            generation_config=mock_genai.GenerationConfig.return_value
        )

class TestOllamaClient(unittest.TestCase):
    @patch('knowledge_storm.lm.dspy.OllamaLocal')
    def test_ollama_client_initialization(self, mock_ollama_local):
        # Create an instance of OllamaClient
        client = OllamaClient(model="llama2", port=11434, url="http://localhost")

        # Check if the client is initialized correctly
        self.assertEqual(client.kwargs['model'], "llama2")

        # Verify that the superclass (dspy.OllamaLocal) was called with the correct parameters
        mock_ollama_local.assert_called_once_with(
            model="llama2",
            base_url="http://localhost:11434"
        )

        # Check if additional kwargs are stored
        self.assertIn('model', client.kwargs)
        self.assertEqual(client.kwargs['model'], "llama2")

        # Verify that the base_url is correctly formatted
        self.assertEqual(mock_ollama_local.call_args[1]['base_url'], "http://localhost:11434")