from knowledge_storm.collaborative_storm.modules.grounded_question_answering import AnswerQuestionModule
from unittest import TestCase
from unittest.mock import Mock, patch

class TestAnswerQuestionModule(TestCase):
    @patch('knowledge_storm.collaborative_storm.modules.grounded_question_answering.dspy')
    @patch('knowledge_storm.collaborative_storm.modules.grounded_question_answering.LoggingWrapper')
    def test_retrieve_information_max_queries(self, mock_logging_wrapper, mock_dspy):
        # Arrange
        max_search_queries = 2
        mock_retriever = Mock()
        mock_lm = Mock()

        # Mock the question_to_query output
        mock_dspy.Predict.return_value.return_value.queries = "- Query 1\n- Query 2\n- Query 3"

        # Create the module
        module = AnswerQuestionModule(
            retriever=mock_retriever,
            max_search_queries=max_search_queries,
            question_answering_lm=mock_lm,
            logging_wrapper=mock_logging_wrapper
        )

        # Act
        queries, searched_results = module.retrieve_information("Test Topic", "Test Question")

        # Assert
        self.assertEqual(len(queries), max_search_queries)
        self.assertEqual(queries, ["Query 1", "Query 2"])
        mock_retriever.retrieve.assert_called_once_with(["Query 1", "Query 2"], exclude_urls=[])
        mock_logging_wrapper.add_query_count.assert_called_once_with(count=2)