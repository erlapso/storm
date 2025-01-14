import dspy
import unittest

from knowledge_storm.collaborative_storm.modules.grounded_question_answering import AnswerQuestionModule
from knowledge_storm.logging_wrapper import LoggingWrapper
from unittest.mock import Mock, patch

class TestAnswerQuestionModule(unittest.TestCase):
    @patch('knowledge_storm.collaborative_storm.modules.grounded_question_answering.dspy.Predict')
    @patch('knowledge_storm.collaborative_storm.modules.grounded_question_answering.dspy.Retrieve')
    def test_no_information_retrieved(self, mock_retrieve, mock_predict):
        # Mock the retriever to return an empty list
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_retrieve.return_value = mock_retriever

        # Mock the QuestionToQuery predict function
        mock_question_to_query = Mock()
        mock_question_to_query.return_value.queries = "- query 1\n- query 2"
        mock_predict.side_effect = [mock_question_to_query, Mock()]

        # Create a mock LoggingWrapper
        mock_logging_wrapper = Mock(spec=LoggingWrapper)

        # Initialize the AnswerQuestionModule with mocked dependencies
        module = AnswerQuestionModule(
            retriever=mock_retriever,
            max_search_queries=2,
            question_answering_lm=Mock(),
            logging_wrapper=mock_logging_wrapper
        )

        # Call the forward method
        result = module.forward(
            topic="Test Topic",
            question="Test Question",
            mode="brief",
            style="conversational"
        )

        # Assert that the response is the default "insufficient information" message
        self.assertEqual(result.response, "Sorry, there is insufficient information to answer the question.")

        # Assert that the raw_retrieved_info is an empty list
        self.assertEqual(result.raw_retrieved_info, [])

        # Assert that the cited_info is an empty dictionary
        self.assertEqual(result.cited_info, {})

        # Assert that the queries were generated
        self.assertEqual(result.queries, ["query 1", "query 2"])

        # Verify that the retrieve method was called
        mock_retriever.retrieve.assert_called_once()

        # Verify that the logging wrapper methods were called
        mock_logging_wrapper.log_event.assert_called()
        mock_logging_wrapper.add_query_count.assert_called_with(count=2)