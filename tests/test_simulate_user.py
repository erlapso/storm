import pytest

from knowledge_storm.collaborative_storm.modules.simulate_user import GenSimulatedUserUtterance
from knowledge_storm.dataclass import ConversationTurn
from unittest.mock import Mock

class TestGenSimulatedUserUtterance:
    @pytest.fixture
    def mock_engine(self):
        return Mock()

    def test_gen_conv_history_string_with_long_history(self, mock_engine):
        # Arrange
        gen_utterance = GenSimulatedUserUtterance(mock_engine)
        conversation_turns = [
            ConversationTurn(role="User", utterance="Hello", claim_to_make=""),
            ConversationTurn(role="AI", utterance="Hi there!", claim_to_make=""),
            ConversationTurn(role="User", utterance="How are you?", claim_to_make=""),
            ConversationTurn(role="AI", utterance="I'm doing well, thank you.", claim_to_make=""),
            ConversationTurn(role="User", utterance="Tell me about AI", claim_to_make="AI is a fascinating field"),
            ConversationTurn(role="AI", utterance="AI, or Artificial Intelligence, is a broad field of computer science...", claim_to_make=""),
        ]

        # Act
        result = gen_utterance.gen_conv_history_string(conversation_turns)

        # Assert
        expected_result = (
            "User: AI is a fascinating field\n"
            "AI: AI, or Artificial Intelligence, is a broad field of computer science...\n"
            "User: Tell me about AI\n"
            "AI: AI, or Artificial Intelligence, is a broad field of computer science..."
        )
        assert result == expected_result, f"Expected:\n{expected_result}\n\nGot:\n{result}"