import dspy
import pytest

from knowledge_storm.collaborative_storm.modules.knowledge_base_summary import KnowledgeBaseSummaryModule
from knowledge_storm.dataclass import KnowledgeBase
from unittest.mock import MagicMock, patch

class TestKnowledgeBaseSummaryModule:
    def test_forward_method_with_empty_knowledge_base(self):
        # Arrange
        mock_engine = MagicMock()
        module = KnowledgeBaseSummaryModule(engine=mock_engine)

        empty_knowledge_base = KnowledgeBase(topic="Empty Topic")

        # Mock the Predict function to return a MagicMock with an output attribute
        mock_predict = MagicMock()
        mock_predict.return_value.output = "Empty summary"

        # Act
        with patch('dspy.Predict', return_value=mock_predict):
            with patch('dspy.settings.context') as mock_context:
                result = module.forward(empty_knowledge_base)

        # Assert
        assert result == "Empty summary"
        mock_predict.assert_called_once_with(topic="Empty Topic", structure="")
        mock_context.assert_called_once_with(lm=mock_engine, show_guidelines=False)