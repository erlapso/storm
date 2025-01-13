import numpy as np
import unittest

from knowledge_storm.encoder import OpenAIEmbeddingModel, get_text_embeddings
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