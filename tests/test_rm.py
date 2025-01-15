import os

from knowledge_storm.rm import YouRM
from unittest import TestCase
from unittest.mock import MagicMock, patch

class TestYouRM(TestCase):
    @patch('knowledge_storm.rm.requests.get')
    def test_yourm_forward_single_query(self, mock_get):
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "url": "https://example.com",
                    "title": "Example Title",
                    "snippet": "Example snippet"
                }
            ]
        }
        mock_get.return_value = mock_response

        # Set up the YouRM instance
        os.environ['YDC_API_KEY'] = 'dummy_key'
        you_rm = YouRM(k=1)

        # Call the forward method
        result = you_rm.forward("test query")

        # Assert the results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['url'], "https://example.com")
        self.assertEqual(result[0]['title'], "Example Title")
        self.assertEqual(result[0]['snippets'], ["Example snippet"])
        self.assertEqual(result[0]['description'], "Example snippet")

        # Assert that the API was called correctly
        mock_get.assert_called_once_with(
            "https://api.ydc-index.io/search?query=test query",
            headers={"X-API-Key": "dummy_key"}
        )

        # Check usage
        self.assertEqual(you_rm.get_usage_and_reset(), {"YouRM": 1})