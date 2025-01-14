import numpy as np
import requests
import unittest

from knowledge_storm.encoder import AzureOpenAIEmbeddingModel, OpenAIEmbeddingModel, get_text_embeddings
from unittest.mock import MagicMock, patch

class TestEncoder(unittest.TestCase):
    @patch.dict('os.environ', {'ENCODER_API_TYPE': 'openai'})
    @patch('knowledge_storm.encoder.OpenAIEmbeddingModel')
    def test_get_text_embeddings_multiple_texts(self, mock_openai_model):
        # Arrange
        mock_model_instance = MagicMock()
        mock_openai_model.return_value = mock_model_instance
        mock_model_instance.get_embedding.side_effect = [
            (np.array([0.1, 0.2, 0.3]), 5),
            (np.array([0.4, 0.5, 0.6]), 7),
            (np.array([0.7, 0.8, 0.9]), 6)
        ]

        texts = ["Hello", "World", "Test"]

        # Act
        embeddings, total_tokens = get_text_embeddings(texts)

        # Assert
        self.assertEqual(embeddings.shape, (3, 3))  # 3 texts, 3-dimensional embeddings
        self.assertEqual(total_tokens, 18)  # Sum of token counts (5 + 7 + 6)

        # Check if the embeddings are in the correct order
        np.testing.assert_array_almost_equal(embeddings[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(embeddings[1], [0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(embeddings[2], [0.7, 0.8, 0.9])

        # Verify that the get_embedding method was called for each text
        self.assertEqual(mock_model_instance.get_embedding.call_count, 3)

    @patch.dict('os.environ', {'ENCODER_API_TYPE': 'openai'})
    @patch('knowledge_storm.encoder.OpenAIEmbeddingModel')
    def test_get_text_embeddings_with_cache(self, mock_openai_model):
        # Arrange
        mock_model_instance = MagicMock()
        mock_openai_model.return_value = mock_model_instance
        mock_model_instance.get_embedding.return_value = (np.array([0.1, 0.2, 0.3]), 5)

        text = "Hello, world!"
        embedding_cache = {}

        # Act
        embeddings1, tokens1 = get_text_embeddings(text, embedding_cache=embedding_cache)
        embeddings2, tokens2 = get_text_embeddings(text, embedding_cache=embedding_cache)

        # Assert
        self.assertEqual(mock_model_instance.get_embedding.call_count, 1)
        np.testing.assert_array_almost_equal(embeddings1, embeddings2)
        self.assertEqual(tokens1, 5)
        self.assertEqual(tokens2, 0)
        self.assertIn(text, embedding_cache)
        np.testing.assert_array_almost_equal(embedding_cache[text], np.array([0.1, 0.2, 0.3]))

    @patch.dict('os.environ', {'ENCODER_API_TYPE': 'invalid_type'})
    def test_get_text_embeddings_invalid_encoder_type(self):
        # Arrange
        text = "Test text"

        # Act & Assert
        with self.assertRaises(Exception) as context:
            get_text_embeddings(text)

        # Check if the error message is as expected
        self.assertIn("No valid encoder type is provided", str(context.exception))

    @patch('knowledge_storm.encoder.AzureOpenAI')
    def test_azure_openai_embedding_model(self, mock_azure_openai):
        # Arrange
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage.prompt_tokens = 5
        mock_client.embeddings.create.return_value = mock_response

        model = AzureOpenAIEmbeddingModel()

        # Act
        embedding, token_count = model.get_embedding("Test text")

        # Assert
        self.assertIsInstance(embedding, np.ndarray)
        np.testing.assert_array_almost_equal(embedding, np.array([0.1, 0.2, 0.3]))
        self.assertEqual(token_count, 5)
        mock_client.embeddings.create.assert_called_once_with(input="Test text", model="text-embedding-3-small")

    @patch.dict('os.environ', {'ENCODER_API_TYPE': 'together'})
    @patch('knowledge_storm.encoder.together')
    def test_together_embedding_model(self, mock_together):
        # Arrange
        mock_client = MagicMock()
        mock_together.Together.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        # Act
        with patch('knowledge_storm.encoder.TogetherEmbeddingModel') as mock_together_model:
            mock_model_instance = mock_together_model.return_value
            mock_model_instance.get_embedding.return_value = (np.array([0.1, 0.2, 0.3]), -1)

            embeddings, total_tokens = get_text_embeddings("Test text")

        # Assert
        self.assertIsInstance(embeddings, np.ndarray)
        np.testing.assert_array_almost_equal(embeddings, np.array([0.1, 0.2, 0.3]))
        self.assertEqual(total_tokens, -1)
        mock_model_instance.get_embedding.assert_called_once_with("Test text")

    @patch('knowledge_storm.encoder.requests.post')
    def test_openai_embedding_model(self, mock_post):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 5}
        }
        mock_post.return_value = mock_response

        custom_api_key = "test_api_key"
        model = OpenAIEmbeddingModel(api_key=custom_api_key)

        # Act
        embedding, token_count = model.get_embedding("Test text")

        # Assert
        self.assertIsInstance(embedding, np.ndarray)
        np.testing.assert_array_almost_equal(embedding, np.array([0.1, 0.2, 0.3]))
        self.assertEqual(token_count, 5)

        mock_post.assert_called_once_with(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {custom_api_key}"
            },
            json={"input": "Test text", "model": "text-embedding-3-small"}
        )

    @patch.dict('os.environ', {'ENCODER_API_TYPE': 'openai'})
    @patch('knowledge_storm.encoder.requests.post')
    def test_get_text_embeddings_api_error(self, mock_post):
        # Arrange
        mock_post.side_effect = requests.RequestException("API Error")
        text = "Test text"

        # Act & Assert
        with self.assertRaises(requests.RequestException) as context:
            get_text_embeddings(text)

        # Check if the error message is as expected
        self.assertIn("API Error", str(context.exception))

        # Verify that the post method was called
        mock_post.assert_called_once()