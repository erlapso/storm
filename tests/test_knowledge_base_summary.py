import dspy
import unittest

from knowledge_storm.collaborative_storm.modules.knowledge_base_summary import KnowledgeBaseSummaryModule
from knowledge_storm.dataclass import KnowledgeBase
from unittest.mock import Mock, patch

class TestKnowledgeBaseSummaryModule(unittest.TestCase):
    def test_forward_method(self):
        # Create a mock KnowledgeBase
        mock_kb = Mock(spec=KnowledgeBase)
        mock_kb.topic = "Test Topic"
        mock_kb.get_node_hierarchy_string.return_value = "# Section 1\n## Subsection 1.1\n# Section 2"

        # Create a mock engine
        mock_engine = Mock()

        # Create a mock Predict function
        mock_predict = Mock()
        mock_predict.return_value.output = "This is a test summary."

        # Patch dspy.Predict to return our mock
        with patch('dspy.Predict', return_value=mock_predict):
            # Create the KnowledgeBaseSummaryModule with the mock engine
            module = KnowledgeBaseSummaryModule(mock_engine)

            # Call the forward method
            summary = module.forward(mock_kb)

        # Assert that the summary is correct
        self.assertEqual(summary, "This is a test summary.")

        # Assert that the mock_predict was called with the correct arguments
        mock_predict.assert_called_once_with(topic="Test Topic", structure="# Section 1\n## Subsection 1.1\n# Section 2")

        # Assert that the get_node_hierarchy_string method was called with the correct arguments
        mock_kb.get_node_hierarchy_string.assert_called_once_with(
            include_indent=False,
            include_full_path=False,
            include_hash_tag=True,
            include_node_content_count=False
        )