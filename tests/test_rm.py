import os
import requests
import unittest

from knowledge_storm.rm import ExaSearch
from unittest.mock import MagicMock, patch

class TestExaSearch(unittest.TestCase):
    @patch('knowledge_storm.rm.Exa')
    def test_exa_search_forward(self, mock_exa):
        # Mock the Exa class and its search_and_contents method
        mock_exa_instance = MagicMock()
        mock_exa.return_value = mock_exa_instance

        # Create a mock response
        mock_result = MagicMock()
        mock_result.title = "Test Title"
        mock_result.url = "https://test.com"
        mock_result.summary = "Test Summary"
        mock_result.score = 0.9

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_exa_instance.search_and_contents.return_value = mock_response

        # Create an instance of ExaSearch
        exa_search = ExaSearch(k=1, exa_api_key="test_api_key")

        # Call the forward method
        results = exa_search.forward("test query")

        # Assert the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], "Test Title")
        self.assertEqual(results[0]['url'], "https://test.com")
        self.assertEqual(results[0]['summary'], "Test Summary")
        self.assertEqual(results[0]['score'], 0.9)

        # Assert that the search_and_contents method was called with the correct parameters
        mock_exa_instance.search_and_contents.assert_called_once_with(
            "test query", type="auto", summary=True, numresults=1
        )

    @patch('knowledge_storm.rm.Exa')
    def test_exa_search_forward_multiple_calls(self, mock_exa):
        # Mock the Exa class and its search_and_contents method
        mock_exa_instance = MagicMock()
        mock_exa.return_value = mock_exa_instance

        # Create mock responses
        mock_result1 = MagicMock(title="Title 1", url="https://test1.com", summary="Summary 1", score=0.9)
        mock_result2 = MagicMock(title="Title 2", url="https://test2.com", summary="Summary 2", score=0.8)
        mock_result3 = MagicMock(title="Title 3", url="https://test3.com", summary="Summary 3", score=0.7)

        mock_response1 = MagicMock()
        mock_response1.results = [mock_result1, mock_result2]

        mock_response2 = MagicMock()
        mock_response2.results = [mock_result3]

        # Set up the side effect to return different responses on subsequent calls
        mock_exa_instance.search_and_contents.side_effect = [mock_response1, mock_response2]

        # Create an instance of ExaSearch with k=3
        exa_search = ExaSearch(k=3, exa_api_key="test_api_key")

        # Call the forward method
        results = exa_search.forward("test query")

        # Assert the results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['title'], "Title 1")
        self.assertEqual(results[1]['title'], "Title 2")
        self.assertEqual(results[2]['title'], "Title 3")

        # Assert that the search_and_contents method was called twice with the correct parameters
        self.assertEqual(mock_exa_instance.search_and_contents.call_count, 2)
        mock_exa_instance.search_and_contents.assert_any_call("test query", type="auto", summary=True, numresults=3)
        mock_exa_instance.search_and_contents.assert_any_call(
            "test query",
            type="auto",
            summary=True,
            num_results=3,
            exclude_domains=["test1.com", "test2.com"]
        )

    @patch('knowledge_storm.rm.Exa')
    def test_exa_search_forward_api_error(self, mock_exa):
        # Mock the Exa class and its search_and_contents method
        mock_exa_instance = MagicMock()
        mock_exa.return_value = mock_exa_instance

        # Set up the search_and_contents method to raise an exception
        mock_exa_instance.search_and_contents.side_effect = requests.RequestException("API Error")

        # Create an instance of ExaSearch
        exa_search = ExaSearch(k=1, exa_api_key="test_api_key")

        # Call the forward method and check for logged error
        with self.assertLogs(level='ERROR') as log:
            results = exa_search.forward("test query")

        # Assert that the results are empty due to the error
        self.assertEqual(results, [])

        # Check that the error was logged
        self.assertIn("Error occurs when searching query test query", log.output[0])

    @patch('knowledge_storm.rm.Exa')
    def test_exa_search_forward_with_exclude_urls_and_fewer_results(self, mock_exa):
        # Mock the Exa class and its search_and_contents method
        mock_exa_instance = MagicMock()
        mock_exa.return_value = mock_exa_instance

        # Create mock results
        mock_result1 = MagicMock(title="Title 1", url="https://test1.com", summary="Summary 1", score=0.9)
        mock_result2 = MagicMock(title="Title 2", url="https://test2.com", summary="Summary 2", score=0.8)
        mock_result3 = MagicMock(title="Title 3", url="https://test3.com", summary="Summary 3", score=0.7)

        # Set up the search_and_contents method to return fewer results than requested
        mock_response = MagicMock()
        mock_response.results = [mock_result1, mock_result2, mock_result3]
        mock_exa_instance.search_and_contents.return_value = mock_response

        # Create an instance of ExaSearch with k=5 (more than the mocked results)
        exa_search = ExaSearch(k=5, exa_api_key="test_api_key")

        # Call the forward method with an exclude_urls parameter
        results = exa_search.forward("test query", exclude_urls=["https://test2.com"])

        # Assert the results
        self.assertEqual(len(results), 2)  # Only 2 results because one is excluded
        self.assertEqual(results[0]['title'], "Title 1")
        self.assertEqual(results[1]['title'], "Title 3")

        # Assert that the search_and_contents method was called with the correct parameters
        mock_exa_instance.search_and_contents.assert_called_once_with(
            "test query", type="auto", summary=True, numresults=5
        )

        # Assert that the excluded URL is not in the results
        self.assertNotIn("https://test2.com", [r['url'] for r in results])

        # Assert that no additional API calls were made to fetch more results
        self.assertEqual(mock_exa_instance.search_and_contents.call_count, 1)

    @patch('knowledge_storm.rm.Exa')
    def test_exa_search_forward_multiple_queries(self, mock_exa):
        # Mock the Exa class and its search_and_contents method
        mock_exa_instance = MagicMock()
        mock_exa.return_value = mock_exa_instance

        # Create mock responses for different queries
        mock_result1 = MagicMock(title="Title 1", url="https://test1.com", summary="Summary 1", score=0.9)
        mock_result2 = MagicMock(title="Title 2", url="https://test2.com", summary="Summary 2", score=0.8)
        mock_result3 = MagicMock(title="Title 3", url="https://test3.com", summary="Summary 3", score=0.7)

        mock_response1 = MagicMock()
        mock_response1.results = [mock_result1]

        mock_response2 = MagicMock()
        mock_response2.results = [mock_result2, mock_result3]

        # Set up the side effect to return different responses for different queries
        mock_exa_instance.search_and_contents.side_effect = [mock_response1, mock_response2]

        # Create an instance of ExaSearch with k=2
        exa_search = ExaSearch(k=2, exa_api_key="test_api_key")

        # Call the forward method with multiple queries
        results = exa_search.forward(["query1", "query2"])

        # Assert the results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['title'], "Title 1")
        self.assertEqual(results[1]['title'], "Title 2")
        self.assertEqual(results[2]['title'], "Title 3")

        # Assert that the search_and_contents method was called twice with the correct parameters
        self.assertEqual(mock_exa_instance.search_and_contents.call_count, 2)
        mock_exa_instance.search_and_contents.assert_any_call("query1", type="auto", summary=True, numresults=2)
        mock_exa_instance.search_and_contents.assert_any_call("query2", type="auto", summary=True, numresults=2)

        # Assert that the usage count is correctly incremented
        self.assertEqual(exa_search.usage, 2)

    @patch('knowledge_storm.rm.Exa')
    @patch.dict(os.environ, {'EXA_API_KEY': 'test_env_api_key'})
    def test_exa_search_init_with_env_variable(self, mock_exa):
        # Mock the Exa class
        mock_exa_instance = MagicMock()
        mock_exa.return_value = mock_exa_instance

        # Create an instance of ExaSearch without providing an API key
        exa_search = ExaSearch(k=1)

        # Assert that the Exa class was instantiated with the API key from the environment variable
        mock_exa.assert_called_once_with(api_key='test_env_api_key')

        # Verify that the exa_api_key attribute is set correctly
        self.assertEqual(exa_search.exa_api_key, 'test_env_api_key')

        # Verify that the exa attribute is set to the mock instance
        self.assertEqual(exa_search.exa, mock_exa_instance)

    def test_exa_search_get_usage_and_reset(self):
        # Create an instance of ExaSearch
        exa_search = ExaSearch(k=1, exa_api_key="test_api_key")

        # Simulate some usage
        exa_search.usage = 5

        # Get the usage and reset
        usage = exa_search.get_usage_and_reset()

        # Assert that the usage is correctly reported
        self.assertEqual(usage, {"ExaSearch": 5})

        # Assert that the usage has been reset to 0
        self.assertEqual(exa_search.usage, 0)

        # Simulate more usage
        exa_search.usage = 3

        # Get the usage and reset again
        usage = exa_search.get_usage_and_reset()

        # Assert that the new usage is correctly reported
        self.assertEqual(usage, {"ExaSearch": 3})

        # Assert that the usage has been reset to 0 again
        self.assertEqual(exa_search.usage, 0)

    @patch.dict(os.environ, {}, clear=True)
    def test_exa_search_init_without_api_key(self):
        # Test initializing ExaSearch without an API key and no environment variable
        with self.assertRaises(RuntimeError) as context:
            ExaSearch(k=1)

        self.assertTrue("You must supply exa_api_key or set environment variable EXA_API_KEY" in str(context.exception))

    @patch('knowledge_storm.rm.Exa')
    def test_exa_search_forward_fewer_results_than_requested(self, mock_exa):
        # Mock the Exa class and its search_and_contents method
        mock_exa_instance = MagicMock()
        mock_exa.return_value = mock_exa_instance

        # Create mock results
        mock_result1 = MagicMock(title="Title 1", url="https://test1.com", summary="Summary 1", score=0.9)
        mock_result2 = MagicMock(title="Title 2", url="https://test2.com", summary="Summary 2", score=0.8)

        # Set up the search_and_contents method to return fewer results than requested
        mock_response = MagicMock()
        mock_response.results = [mock_result1, mock_result2]
        mock_exa_instance.search_and_contents.return_value = mock_response

        # Create an instance of ExaSearch with k=5 (more than the mocked results)
        exa_search = ExaSearch(k=5, exa_api_key="test_api_key")

        # Call the forward method
        results = exa_search.forward("test query")

        # Assert the results
        self.assertEqual(len(results), 2)  # Only 2 results available
        self.assertEqual(results[0]['title'], "Title 1")
        self.assertEqual(results[1]['title'], "Title 2")

        # Assert that the search_and_contents method was called twice
        self.assertEqual(mock_exa_instance.search_and_contents.call_count, 2)

        # Assert that the second call included the exclude_domains parameter
        mock_exa_instance.search_and_contents.assert_any_call(
            "test query",
            type="auto",
            summary=True,
            num_results=5,
            exclude_domains=["test1.com", "test2.com"]
        )

        # Assert that no more calls were made after the second attempt
        self.assertEqual(mock_exa_instance.search_and_contents.call_count, 2)

    @patch('knowledge_storm.rm.Exa')
    def test_exa_search_forward_empty_query_list(self, mock_exa):
        # Mock the Exa class and its search_and_contents method
        mock_exa_instance = MagicMock()
        mock_exa.return_value = mock_exa_instance

        # Create an instance of ExaSearch
        exa_search = ExaSearch(k=3, exa_api_key="test_api_key")

        # Call the forward method with an empty list of queries
        results = exa_search.forward([])

        # Assert that the results are empty
        self.assertEqual(results, [])

        # Assert that search_and_contents was not called
        mock_exa_instance.search_and_contents.assert_not_called()

        # Assert that the usage count is 0
        self.assertEqual(exa_search.usage, 0)