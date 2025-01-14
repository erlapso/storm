import unittest

from knowledge_storm.collaborative_storm.modules.grounded_question_generation import GroundedQuestionGenerationModule
from knowledge_storm.dataclass import ConversationTurn, KnowledgeBase
from knowledge_storm.interface import Information
from unittest.mock import MagicMock, patch

class TestGroundedQuestionGenerationModule(unittest.TestCase):

    def setUp(self):
        self.mock_engine = MagicMock()
        self.module = GroundedQuestionGenerationModule(self.mock_engine)

    @patch('knowledge_storm.collaborative_storm.modules.grounded_question_generation.dspy')
    def test_generate_grounded_question(self, mock_dspy):
        # Arrange
        topic = "Artificial Intelligence"
        knowledge_base = MagicMock(spec=KnowledgeBase)
        knowledge_base.get_knowledge_base_summary.return_value = "AI has made significant progress in recent years."
        last_conv_turn = ConversationTurn(role="Expert", utterance="AI has many applications in healthcare.")
        unused_snippets = [Information(content="AI is also used in finance for fraud detection.")]

        mock_dspy.Prediction.return_value = MagicMock(
            raw_utterance="How is AI being applied in the financial sector, particularly in fraud detection?[1]",
            utterance="Building on the discussion of AI in healthcare, I'm curious about its applications in other industries. How is AI being applied in the financial sector, particularly in fraud detection?[1]",
            cited_info=[unused_snippets[0]]
        )

        # Act
        result = self.module.forward(topic, knowledge_base, last_conv_turn, unused_snippets)

        # Assert
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, MagicMock))  # Since we mocked dspy.Prediction
        self.assertIn("financial sector", result.utterance)
        self.assertIn("fraud detection", result.utterance)
        self.assertEqual(result.cited_info, [unused_snippets[0]])

        # Verify that the module methods were called with correct arguments
        mock_dspy.Predict.assert_called()
        knowledge_base.get_knowledge_base_summary.assert_called_once()

if __name__ == '__main__':
    unittest.main()