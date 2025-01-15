import dspy
import unittest

from knowledge_storm.collaborative_storm.modules.simulate_user import GenSimulatedUserUtterance
from knowledge_storm.dataclass import ConversationTurn
from unittest.mock import MagicMock, patch

class TestGenSimulatedUserUtterance(unittest.TestCase):
    @patch('knowledge_storm.collaborative_storm.modules.simulate_user.AskQuestionWithPersona')
    @patch('dspy.Predict')
    def test_forward_method(self, mock_predict, mock_ask_question):
        # Create a mock engine
        mock_engine = MagicMock()

        # Create an instance of GenSimulatedUserUtterance
        gen_utterance = GenSimulatedUserUtterance(mock_engine)

        # Mock the return value of ask_question
        mock_ask_question.return_value.question = "Mocked question"

        # Create sample conversation turns
        conv_turns = [
            ConversationTurn(role="user", utterance="Hello"),
            ConversationTurn(role="assistant", utterance="Hi there"),
            ConversationTurn(role="user", utterance="Tell me about AI"),
        ]

        # Call the forward method
        with patch.object(dspy.settings, 'context') as mock_context:
            result = gen_utterance.forward("AI", "latest developments", conv_turns)

        # Assert that the result is the mocked question
        self.assertEqual(result, "Mocked question")

        # Assert that the ask_question method was called with the correct arguments
        mock_predict.return_value.assert_called_once_with(
            topic="AI",
            persona="researcher with interest in latest developments",
            conv=gen_utterance.gen_conv_history_string(conv_turns)
        )

        # Assert that the context manager was used
        mock_context.assert_called_once_with(lm=mock_engine, show_guidelines=False)

if __name__ == '__main__':
    unittest.main()