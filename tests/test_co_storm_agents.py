import pytest

from knowledge_storm.collaborative_storm.engine import RunnerArgument
from knowledge_storm.collaborative_storm.modules.callback import BaseCallbackHandler
from knowledge_storm.collaborative_storm.modules.co_storm_agents import SimulatedUser
from knowledge_storm.dataclass import ConversationTurn, KnowledgeBase
from knowledge_storm.interface import LMConfigs
from knowledge_storm.logging_wrapper import LoggingWrapper
from unittest import mock

def test_simulated_user_generate_utterance():
    # Set up the SimulatedUser instance
    topic = "AI Ethics"
    role_name = "Simulated User"
    role_description = "A simulated user interested in AI ethics"
    intent = "Learn about AI ethics implications"
    lm_config = mock.Mock(spec=LMConfigs)
    runner_argument = mock.Mock(spec=RunnerArgument)
    logging_wrapper = mock.Mock(spec=LoggingWrapper)
    callback_handler = mock.Mock(spec=BaseCallbackHandler)

    simulated_user = SimulatedUser(
        topic=topic,
        role_name=role_name,
        role_description=role_description,
        intent=intent,
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        callback_handler=callback_handler
    )

    # Mock the knowledge base
    knowledge_base = mock.Mock(spec=KnowledgeBase)

    # Create a mock conversation history
    conversation_history = [
        ConversationTurn(role="Expert", raw_utterance="AI ethics involves considering the ethical implications of AI systems.", utterance_type="Potential Answer")
    ]

    # Mock the gen_simulated_user_utterance method
    simulated_user.gen_simulated_user_utterance = mock.Mock()
    simulated_user.gen_simulated_user_utterance.return_value = "What are some specific ethical concerns in AI development?"

    # Call the generate_utterance method
    result = simulated_user.generate_utterance(knowledge_base, conversation_history)

    # Assert that the method calls were made correctly
    simulated_user.gen_simulated_user_utterance.assert_called_once_with(
        topic=topic,
        intent=intent,
        conv_history=conversation_history
    )

    # Assert that the result is a ConversationTurn with the expected attributes
    assert isinstance(result, ConversationTurn)
    assert result.role == "Guest"
    assert result.raw_utterance == "What are some specific ethical concerns in AI development?"
    assert result.utterance_type == "Original Question"

    # Test that the method raises an assertion error when intent is not set
    simulated_user.intent = None
    with pytest.raises(AssertionError):
        simulated_user.generate_utterance(knowledge_base, conversation_history)