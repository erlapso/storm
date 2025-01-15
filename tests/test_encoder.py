import numpy as np
import unittest

from knowledge_storm.encoder import OpenAIEmbeddingModel, get_text_embeddings
from unittest.mock import MagicMock, patch

class TestEncoder(unittest.TestCase):
    @patch.dict('os.environ', {'ENCODER_API_TYPE': 'openai'})
    @patch('knowledge_storm.encoder.OpenAIEmbeddingModel')
    def test_get_text_embeddings_list(self, mock_openai_model):
        # Mock the OpenAIEmbeddingModel
        mock_model_instance = MagicMock()
        mock_openai_model.return_value = mock_model_instance

        # Set up mock return values for get_embedding
        mock_embeddings = [
            (np.array([0.1, 0.2, 0.3]), 10),
            (np.array([0.4, 0.5, 0.6]), 15),
            (np.array([0.7, 0.8, 0.9]), 20)
        ]
        mock_model_instance.get_embedding.side_effect = mock_embeddings

        # Test input
        texts = ["Hello", "World", "Test"]

        # Call the function
        embeddings, total_tokens = get_text_embeddings(texts)

        # Assertions
        self.assertEqual(mock_model_instance.get_embedding.call_count, 3)
        self.assertEqual(embeddings.shape, (3, 3))
        self.assertEqual(total_tokens, 45)
        np.testing.assert_array_equal(embeddings[0], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(embeddings[1], np.array([0.4, 0.5, 0.6]))
        np.testing.assert_array_equal(embeddings[2], np.array([0.7, 0.8, 0.9]))

if __name__ == '__main__':
    unittest.main()