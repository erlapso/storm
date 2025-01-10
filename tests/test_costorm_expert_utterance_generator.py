import dspy
import pytest
import unittest

from knowledge_storm.collaborative_storm.modules.collaborative_storm_utils import extract_and_remove_citations, keep_first_and_last_paragraph
from knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator import CoStormExpertUtteranceGenerationModule
from knowledge_storm.dataclass import ConversationTurn
from unittest.mock import Mock, patch

class TestCoStormExpertUtteranceGenerationModule(unittest.TestCase):
    def setUp(self):
        # Mock the dependencies
        self.mock_action_planning_lm = Mock()
        self.mock_utterance_polishing_lm = Mock()
        self.mock_answer_question_module = Mock()
        self.mock_logging_wrapper = Mock()

        # Create an instance of the module
        self.module = CoStormExpertUtteranceGenerationModule(
            action_planning_lm=self.mock_action_planning_lm,
            utterance_polishing_lm=self.mock_utterance_polishing_lm,
            answer_question_module=self.mock_answer_question_module,
            logging_wrapper=self.mock_logging_wrapper
        )

    def test_forward_with_information_request(self):
        # Arrange
        topic = "AI Ethics"
        current_expert = "AI Researcher"
        conversation_summary = "Discussion about AI ethics and its implications."
        last_conv_turn = ConversationTurn(
            role="Philosopher",
            utterance="What are the main ethical concerns in AI development?",
            utterance_type="Information Request"
        )

        # Mock the answer_question_module response
        mock_answer = Mock()
        mock_answer.response = "The main ethical concerns in AI development include bias, privacy, and accountability."
        mock_answer.queries = ["AI ethics concerns"]
        mock_answer.raw_retrieved_info = "Raw info about AI ethics"
        mock_answer.cited_info = "Cited info about AI ethics"
        self.mock_answer_question_module.return_value = mock_answer

        # Act
        with patch.object(dspy.settings, 'context') as mock_context:
            result = self.module.forward(topic, current_expert, conversation_summary, last_conv_turn)

        # Assert
        self.assertEqual(result.conversation_turn.role, current_expert)
        self.assertEqual(result.conversation_turn.utterance_type, "Potential Answer")
        self.assertEqual(result.conversation_turn.raw_utterance, mock_answer.response)
        self.assertEqual(result.conversation_turn.queries, mock_answer.queries)
        self.assertEqual(result.conversation_turn.raw_retrieved_info, mock_answer.raw_retrieved_info)
        self.assertEqual(result.conversation_turn.cited_info, mock_answer.cited_info)

        # Verify that the answer_question_module was called with the correct arguments
        self.mock_answer_question_module.assert_called_once_with(
            topic=topic,
            question="What are the main ethical concerns in AI development?",
            mode="brief",
            style="conversational and concise",
            callback_handler=None
        )

        # Verify that the context manager for dspy settings was used
        mock_context.assert_called_once_with(lm=self.mock_action_planning_lm, show_guidelines=False)

    def test_forward_with_original_question(self):
        # Arrange
        topic = "Climate Change"
        current_expert = "Environmental Scientist"
        conversation_summary = "Discussion about the impacts of climate change."
        last_conv_turn = ConversationTurn(
            role="Economist",
            utterance="The economic impacts of climate change are significant.",
            utterance_type="Further Details"
        )

        # Mock the action_planning_lm to return an "Original Question" action
        mock_action = Mock()
        mock_action.response = "Original Question: How do different ecosystems respond to rising temperatures?"
        self.module.expert_action = Mock(return_value=mock_action)

        # Act
        with patch.object(dspy.settings, 'context') as mock_context:
            result = self.module.forward(topic, current_expert, conversation_summary, last_conv_turn)

        # Assert
        self.assertEqual(result.conversation_turn.role, current_expert)
        self.assertEqual(result.conversation_turn.utterance_type, "Original Question")
        self.assertEqual(result.conversation_turn.raw_utterance, "How do different ecosystems respond to rising temperatures?")

        # Verify that the action_planning_lm was called with the correct arguments
        self.module.expert_action.assert_called_once_with(
            topic=topic,
            expert=current_expert,
            summary=conversation_summary,
            last_utterance="The economic impacts of climate change are significant."
        )

        # Verify that the answer_question_module was not called
        self.mock_answer_question_module.assert_not_called()

        # Verify that the context manager for dspy settings was used
        mock_context.assert_called_once_with(lm=self.mock_action_planning_lm, show_guidelines=False)

    def test_forward_with_undefined_action(self):
        # Arrange
        topic = "Quantum Computing"
        current_expert = "Quantum Physicist"
        conversation_summary = "Discussion about recent advancements in quantum computing."
        last_conv_turn = ConversationTurn(
            role="Computer Scientist",
            utterance="Classical computers are reaching their limits in certain areas.",
            utterance_type="Further Details"
        )

        # Mock the action_planning_lm to return an undefined action
        mock_action = Mock()
        mock_action.response = "Unexpected Action: This is not a valid action type"
        self.module.expert_action = Mock(return_value=mock_action)

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            with patch.object(dspy.settings, 'context'):
                self.module.forward(topic, current_expert, conversation_summary, last_conv_turn)

        # Assert the exception message
        assert str(exc_info.value) == "unexpected output: Unexpected Action: This is not a valid action type"

        # Verify that the expert_action was called with the correct arguments
        self.module.expert_action.assert_called_once_with(
            topic=topic,
            expert=current_expert,
            summary=conversation_summary,
            last_utterance="Classical computers are reaching their limits in certain areas."
        )

        # Verify that the answer_question_module was not called
        self.mock_answer_question_module.assert_not_called()

    def test_forward_with_potential_answer(self):
        # Arrange
        topic = "Space Exploration"
        current_expert = "Astronaut"
        conversation_summary = "Discussion about future Mars missions."
        last_conv_turn = ConversationTurn(
            role="Astrophysicist",
            utterance="What are the main challenges for long-term human habitation on Mars?",
            utterance_type="Original Question"
        )

        # Mock the answer_question_module response
        mock_answer = Mock()
        mock_answer.response = "The main challenges for long-term human habitation on Mars include radiation exposure, low gravity effects, and resource management."
        mock_answer.queries = ["Mars habitation challenges"]
        mock_answer.raw_retrieved_info = "Raw info about Mars habitation"
        mock_answer.cited_info = "Cited info about Mars habitation"
        self.mock_answer_question_module.return_value = mock_answer

        # Act
        with patch.object(dspy.settings, 'context') as mock_context:
            result = self.module.forward(topic, current_expert, conversation_summary, last_conv_turn)

        # Assert
        self.assertEqual(result.conversation_turn.role, current_expert)
        self.assertEqual(result.conversation_turn.utterance_type, "Potential Answer")
        self.assertEqual(result.conversation_turn.raw_utterance, mock_answer.response)
        self.assertEqual(result.conversation_turn.queries, mock_answer.queries)
        self.assertEqual(result.conversation_turn.raw_retrieved_info, mock_answer.raw_retrieved_info)
        self.assertEqual(result.conversation_turn.cited_info, mock_answer.cited_info)
        self.assertEqual(result.conversation_turn.claim_to_make, "What are the main challenges for long-term human habitation on Mars?")

        # Verify that the answer_question_module was called with the correct arguments
        self.mock_answer_question_module.assert_called_once_with(
            topic=topic,
            question="What are the main challenges for long-term human habitation on Mars?",
            mode="brief",
            style="conversational and concise",
            callback_handler=None
        )

        # Verify that the context manager for dspy settings was used
        mock_context.assert_called_once_with(lm=self.mock_utterance_polishing_lm, show_guidelines=False)

        # Verify that expert_action was not called
        self.module.expert_action.assert_not_called()

    def test_polish_utterance(self):
        # Arrange
        current_turn = ConversationTurn(
            role="AI Researcher",
            utterance_type="Further Details",
            claim_to_make="The impact of transformer models on NLP tasks",
            raw_utterance="Transformer models have revolutionized NLP."
        )
        last_turn = ConversationTurn(
            role="Tech Journalist",
            utterance="AI has made significant progress in natural language processing. [1]",
            utterance_type="Further Details"
        )

        # Mock the change_style method
        mock_change_style = Mock()
        mock_change_style.return_value.utterance = "As an AI Researcher, I can elaborate on the impact of transformer models. They have indeed revolutionized NLP by enabling better understanding of context."
        self.module.change_style = mock_change_style

        # Act
        with patch.object(dspy.settings, 'context') as mock_context:
            self.module.polish_utterance(current_turn, last_turn)

        # Assert
        self.assertEqual(current_turn.utterance, mock_change_style.return_value.utterance)

        # Verify that change_style was called with the correct arguments
        mock_change_style.assert_called_once_with(
            expert="AI Researcher",
            action="Further Details about: The impact of transformer models on NLP tasks",
            prev="AI has made significant progress in natural language processing.",
            content="Transformer models have revolutionized NLP."
        )

        # Verify that the context manager for dspy settings was used
        mock_context.assert_called_once_with(lm=self.module.utterance_polishing_lm, show_guidelines=False)

        # Verify that extract_and_remove_citations and keep_first_and_last_paragraph were called
        with patch('knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator.extract_and_remove_citations') as mock_extract:
            with patch('knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator.keep_first_and_last_paragraph') as mock_keep:
                mock_extract.return_value = ("AI has made significant progress in natural language processing.", None)
                mock_keep.return_value = "AI has made significant progress in natural language processing."
                self.module.polish_utterance(current_turn, last_turn)
                mock_extract.assert_called_once()
                mock_keep.assert_called_once()

    def test_parse_action(self):
        # Arrange
        module = CoStormExpertUtteranceGenerationModule(
            action_planning_lm=Mock(),
            utterance_polishing_lm=Mock(),
            answer_question_module=Mock(),
            logging_wrapper=Mock()
        )

        # Test cases
        test_cases = [
            ("Original Question: What is the impact of AI on job markets?", "Original Question", "What is the impact of AI on job markets?"),
            ("[Further Details]: AI has both positive and negative effects on employment.", "Further Details", "AI has both positive and negative effects on employment."),
            ("Information Request: Can you provide statistics on AI-related job creation?", "Information Request", "Can you provide statistics on AI-related job creation?"),
            ("[Potential Answer]: AI is likely to create new job categories while automating others.", "Potential Answer", "AI is likely to create new job categories while automating others."),
            ("Undefined action type: This should return Undefined", "Undefined", ""),
        ]

        # Act and Assert
        for action, expected_type, expected_content in test_cases:
            action_type, action_content = module.parse_action(action)
            self.assertEqual(action_type, expected_type)
            self.assertEqual(action_content, expected_content)

    def test_forward_with_no_callback_handler(self):
        # Arrange
        topic = "Artificial Intelligence"
        current_expert = "AI Ethicist"
        conversation_summary = "Discussion about ethical implications of AI."
        last_conv_turn = ConversationTurn(
            role="Data Scientist",
            utterance="AI models can sometimes produce biased results.",
            utterance_type="Further Details"
        )

        # Create a module instance with no callback handler
        module_without_callback = CoStormExpertUtteranceGenerationModule(
            action_planning_lm=self.mock_action_planning_lm,
            utterance_polishing_lm=self.mock_utterance_polishing_lm,
            answer_question_module=self.mock_answer_question_module,
            logging_wrapper=self.mock_logging_wrapper,
            callback_handler=None
        )

        # Mock the action_planning_lm to return a "Further Details" action
        mock_action = Mock()
        mock_action.response = "Further Details: The importance of ethical considerations in AI development"
        module_without_callback.expert_action = Mock(return_value=mock_action)

        # Mock the answer_question_module response
        mock_answer = Mock()
        mock_answer.response = "Ethical considerations are crucial in AI development to ensure fairness and prevent harm."
        mock_answer.queries = ["AI ethics importance"]
        mock_answer.raw_retrieved_info = "Raw info about AI ethics"
        mock_answer.cited_info = "Cited info about AI ethics"
        self.mock_answer_question_module.return_value = mock_answer

        # Act
        with patch.object(dspy.settings, 'context'):
            result = module_without_callback.forward(topic, current_expert, conversation_summary, last_conv_turn)

        # Assert
        self.assertEqual(result.conversation_turn.role, current_expert)
        self.assertEqual(result.conversation_turn.utterance_type, "Further Details")
        self.assertEqual(result.conversation_turn.raw_utterance, mock_answer.response)

        # Verify that the expert_action was called with the correct arguments
        module_without_callback.expert_action.assert_called_once_with(
            topic=topic,
            expert=current_expert,
            summary=conversation_summary,
            last_utterance="AI models can sometimes produce biased results."
        )

        # Verify that the answer_question_module was called with callback_handler=None
        self.mock_answer_question_module.assert_called_once_with(
            topic=topic,
            question="The importance of ethical considerations in AI development",
            mode="brief",
            style="conversational and concise",
            callback_handler=None
        )

    def test_polish_utterance_for_different_action_types(self):
        # Arrange
        module = CoStormExpertUtteranceGenerationModule(
            action_planning_lm=Mock(),
            utterance_polishing_lm=Mock(),
            answer_question_module=Mock(),
            logging_wrapper=Mock()
        )

        # Mock the change_style method
        mock_change_style = Mock()
        mock_change_style.return_value.utterance = "Polished utterance"
        module.change_style = mock_change_style

        # Mock extract_and_remove_citations and keep_first_and_last_paragraph
        with patch('knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator.extract_and_remove_citations') as mock_extract:
            with patch('knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator.keep_first_and_last_paragraph') as mock_keep:
                mock_extract.return_value = ("Last utterance without citations", None)
                mock_keep.return_value = "Trimmed last utterance"

                # Test cases for different action types
                test_cases = [
                    ("Original Question", "What is the impact of AI?"),
                    ("Information Request", "Can you provide more details on AI ethics?"),
                    ("Further Details", "AI has both positive and negative effects"),
                    ("Potential Answer", "AI is likely to create new job categories")
                ]

                for action_type, claim_to_make in test_cases:
                    # Arrange
                    current_turn = ConversationTurn(
                        role="AI Expert",
                        utterance_type=action_type,
                        claim_to_make=claim_to_make,
                        raw_utterance="Initial raw utterance"
                    )
                    last_turn = ConversationTurn(
                        role="Previous Expert",
                        utterance="Last expert's utterance",
                        utterance_type="Further Details"
                    )

                    # Act
                    with patch.object(dspy.settings, 'context'):
                        module.polish_utterance(current_turn, last_turn)

                    # Assert
                    self.assertEqual(current_turn.utterance, "Polished utterance")

                    # Verify that change_style was called with the correct arguments
                    expected_action = f"{action_type}"
                    if action_type in ["Further Details", "Potential Answer"]:
                        expected_action += f" about: {claim_to_make}"

                    mock_change_style.assert_called_with(
                        expert="AI Expert",
                        action=expected_action,
                        prev="Trimmed last utterance",
                        content="Initial raw utterance"
                    )

                    # Reset the mock for the next iteration
                    mock_change_style.reset_mock()

    def test_polish_utterance_for_different_action_types(self):
        # Arrange
        module = CoStormExpertUtteranceGenerationModule(
            action_planning_lm=Mock(),
            utterance_polishing_lm=Mock(),
            answer_question_module=Mock(),
            logging_wrapper=Mock()
        )

        # Mock the change_style method
        mock_change_style = Mock()
        mock_change_style.return_value.utterance = "Polished utterance"
        module.change_style = mock_change_style

        # Mock extract_and_remove_citations and keep_first_and_last_paragraph
        with patch('knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator.extract_and_remove_citations') as mock_extract:
            with patch('knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator.keep_first_and_last_paragraph') as mock_keep:
                mock_extract.return_value = ("Last utterance without citations", None)
                mock_keep.return_value = "Trimmed last utterance"

                # Test cases for different action types
                test_cases = [
                    ("Original Question", "What is the impact of AI?"),
                    ("Information Request", "Can you provide more details on AI ethics?"),
                    ("Further Details", "AI has both positive and negative effects"),
                    ("Potential Answer", "AI is likely to create new job categories")
                ]

                for action_type, claim_to_make in test_cases:
                    # Arrange
                    current_turn = ConversationTurn(
                        role="AI Expert",
                        utterance_type=action_type,
                        claim_to_make=claim_to_make,
                        raw_utterance="Initial raw utterance"
                    )
                    last_turn = ConversationTurn(
                        role="Previous Expert",
                        utterance="Last expert's utterance",
                        utterance_type="Further Details"
                    )

                    # Act
                    with patch.object(dspy.settings, 'context'):
                        module.polish_utterance(current_turn, last_turn)

                    # Assert
                    self.assertEqual(current_turn.utterance, "Polished utterance")

                    # Verify that change_style was called with the correct arguments
                    expected_action = f"{action_type}"
                    if action_type in ["Further Details", "Potential Answer"]:
                        expected_action += f" about: {claim_to_make}"

                    mock_change_style.assert_called_with(
                        expert="AI Expert",
                        action=expected_action,
                        prev="Trimmed last utterance",
                        content="Initial raw utterance"
                    )

                    # Reset the mock for the next iteration
                    mock_change_style.reset_mock()

    def test_polish_utterance_for_different_action_types(self):
        # Arrange
        module = CoStormExpertUtteranceGenerationModule(
            action_planning_lm=Mock(),
            utterance_polishing_lm=Mock(),
            answer_question_module=Mock(),
            logging_wrapper=Mock()
        )

        # Mock the change_style method
        mock_change_style = Mock()
        mock_change_style.return_value.utterance = "Polished utterance"
        module.change_style = mock_change_style

        # Mock extract_and_remove_citations and keep_first_and_last_paragraph
        with patch('knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator.extract_and_remove_citations') as mock_extract:
            with patch('knowledge_storm.collaborative_storm.modules.costorm_expert_utterance_generator.keep_first_and_last_paragraph') as mock_keep:
                mock_extract.return_value = ("Last utterance without citations", None)
                mock_keep.return_value = "Trimmed last utterance"

                # Test cases for different action types
                test_cases = [
                    ("Original Question", "What is the impact of AI?"),
                    ("Information Request", "Can you provide more details on AI ethics?"),
                    ("Further Details", "AI has both positive and negative effects"),
                    ("Potential Answer", "AI is likely to create new job categories")
                ]

                for action_type, claim_to_make in test_cases:
                    # Arrange
                    current_turn = ConversationTurn(
                        role="AI Expert",
                        utterance_type=action_type,
                        claim_to_make=claim_to_make,
                        raw_utterance="Initial raw utterance"
                    )
                    last_turn = ConversationTurn(
                        role="Previous Expert",
                        utterance="Last expert's utterance",
                        utterance_type="Further Details"
                    )

                    # Act
                    with patch.object(dspy.settings, 'context'):
                        module.polish_utterance(current_turn, last_turn)

                    # Assert
                    self.assertEqual(current_turn.utterance, "Polished utterance")

                    # Verify that change_style was called with the correct arguments
                    expected_action = f"{action_type}"
                    if action_type in ["Further Details", "Potential Answer"]:
                        expected_action += f" about: {claim_to_make}"

                    mock_change_style.assert_called_with(
                        expert="AI Expert",
                        action=expected_action,
                        prev="Trimmed last utterance",
                        content="Initial raw utterance"
                    )

                    # Reset the mock for the next iteration
                    mock_change_style.reset_mock()