import io
import json
import logging
import os
import requests
import unittest

from knowledge_storm.rm import YouRM
from unittest.mock import MagicMock, patch

class TestYouRM(unittest.TestCase):
    def test_initialization(self):
        # Test initialization with API key provided
        rm = YouRM(ydc_api_key="test_key")
        self.assertEqual(rm.ydc_api_key, "test_key")

        # Test initialization with environment variable
        with patch.dict(os.environ, {"YDC_API_KEY": "env_key"}):
            rm = YouRM()
            self.assertEqual(rm.ydc_api_key, "env_key")

        # Test initialization without API key (should raise an exception)
        with self.assertRaises(RuntimeError):
            YouRM()

    def test_get_usage_and_reset(self):
        rm = YouRM(ydc_api_key="test_key")

        # Simulate some usage
        rm.usage = 5

        # Check usage and reset
        usage = rm.get_usage_and_reset()
        self.assertEqual(usage, {"YouRM": 5})

        # Check if usage was reset
        self.assertEqual(rm.usage, 0)

        # Check usage again (should be 0)
        usage = rm.get_usage_and_reset()
        self.assertEqual(usage, {"YouRM": 0})

    @patch('requests.get')
    def test_forward(self, mock_get):
        # Prepare mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "url": "https://example.com",
                    "title": "Example Title",
                    "snippet": "Example Snippet"
                }
            ]
        }
        mock_get.return_value = mock_response

        # Initialize YouRM
        rm = YouRM(ydc_api_key="test_key")

        # Call forward method
        results = rm.forward("test query")

        # Assert the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['url'], "https://example.com")
        self.assertEqual(results[0]['title'], "Example Title")
        self.assertEqual(results[0]['snippets'], ["Example Snippet"])

        # Check if the API was called with correct parameters
        mock_get.assert_called_once_with(
            "https://api.ydc-index.io/search?query=test query",
            headers={"X-API-Key": "test_key"}
        )

        # Check if usage was incremented
        self.assertEqual(rm.usage, 1)

    @patch('requests.get')
    def test_forward_with_is_valid_source(self, mock_get):
        # Prepare mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "url": "https://example.com",
                    "title": "Example Title",
                    "snippet": "Example Snippet"
                },
                {
                    "url": "https://invalid.com",
                    "title": "Invalid Title",
                    "snippet": "Invalid Snippet"
                }
            ]
        }
        mock_get.return_value = mock_response

        # Define a custom is_valid_source function
        def is_valid_source(url):
            return "example.com" in url

        # Initialize YouRM with custom is_valid_source
        rm = YouRM(ydc_api_key="test_key", is_valid_source=is_valid_source)

        # Call forward method
        results = rm.forward("test query")

        # Assert the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['url'], "https://example.com")
        self.assertEqual(results[0]['title'], "Example Title")
        self.assertEqual(results[0]['snippets'], ["Example Snippet"])

        # Check if the API was called with correct parameters
        mock_get.assert_called_once_with(
            "https://api.ydc-index.io/search?query=test query",
            headers={"X-API-Key": "test_key"}
        )

        # Check if usage was incremented
        self.assertEqual(rm.usage, 1)

    @patch('requests.get')
    def test_forward_multiple_queries(self, mock_get):
        # Prepare mock responses for two queries
        mock_responses = [
            MagicMock(json=lambda: {"hits": [{"url": "https://example1.com", "title": "Example 1", "snippet": "Snippet 1"}]}),
            MagicMock(json=lambda: {"hits": [{"url": "https://example2.com", "title": "Example 2", "snippet": "Snippet 2"}]})
        ]
        mock_get.side_effect = mock_responses

        # Initialize YouRM
        rm = YouRM(ydc_api_key="test_key")

        # Call forward method with multiple queries
        results = rm.forward(["query1", "query2"])

        # Assert the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['url'], "https://example1.com")
        self.assertEqual(results[0]['title'], "Example 1")
        self.assertEqual(results[0]['snippets'], ["Snippet 1"])
        self.assertEqual(results[1]['url'], "https://example2.com")
        self.assertEqual(results[1]['title'], "Example 2")
        self.assertEqual(results[1]['snippets'], ["Snippet 2"])

        # Check if the API was called twice with correct parameters
        self.assertEqual(mock_get.call_count, 2)
        mock_get.assert_any_call(
            "https://api.ydc-index.io/search?query=query1",
            headers={"X-API-Key": "test_key"}
        )
        mock_get.assert_any_call(
            "https://api.ydc-index.io/search?query=query2",
            headers={"X-API-Key": "test_key"}
        )

        # Check if usage was incremented correctly
        self.assertEqual(rm.usage, 2)

    @patch('requests.get')
    def test_forward_with_exception(self, mock_get):
        # Set up mock to raise an exception
        mock_get.side_effect = Exception("Test exception")

        # Set up logging capture
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logging.getLogger().addHandler(handler)

        # Initialize YouRM
        rm = YouRM(ydc_api_key="test_key")

        # Call forward method
        results = rm.forward("test query")

        # Assert the results
        self.assertEqual(results, [])

        # Check if the error was logged
        log_contents = log_capture.getvalue()
        self.assertIn("Error occurs when searching query test query: Test exception", log_contents)

        # Check if usage was incremented despite the exception
        self.assertEqual(rm.usage, 1)

        # Clean up logging
        logging.getLogger().removeHandler(handler)

    @patch('requests.get')
    def test_forward_with_exclude_urls(self, mock_get):
        # Prepare mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "url": "https://example.com",
                    "title": "Example Title",
                    "snippet": "Example Snippet"
                },
                {
                    "url": "https://exclude.com",
                    "title": "Exclude Title",
                    "snippet": "Exclude Snippet"
                },
                {
                    "url": "https://another.com",
                    "title": "Another Title",
                    "snippet": "Another Snippet"
                }
            ]
        }
        mock_get.return_value = mock_response