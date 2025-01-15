from knowledge_storm.collaborative_storm.modules.warmstart_hierarchical_chat import WarmStartModule
from knowledge_storm.dataclass import ConversationTurn, KnowledgeBase
from knowledge_storm.interface import LMConfigs
from knowledge_storm.logging_wrapper import LoggingWrapper
from typing import List
from unittest import TestCase, mock

class TestWarmStartModule(TestCase):
    @mock.patch('knowledge_storm.collaborative_storm.modules.warmstart_hierarchical_chat.GenerateExpertModule')
    @mock.patch('knowledge_storm.collaborative_storm.modules.warmstart_hierarchical_chat.WarmStartConversation')
    @mock.patch('knowledge_storm.collaborative_storm.modules.warmstart_hierarchical_chat.GenerateWarmStartOutlineModule')
    @mock.patch('knowledge_storm.collaborative_storm.modules.warmstart_hierarchical_chat.ReportToConversation')
    def test_initiate_warm_start(self, mock_report_to_conversation, mock_generate_warmstart_outline_module, 
                                 mock_warmstart_conversation, mock_generate_expert_module):
        # Mock LMConfigs and RunnerArgument
        mock_lm_config = mock.MagicMock(spec=LMConfigs)
        mock_runner_argument = mock.MagicMock()
        mock_logging_wrapper = mock.MagicMock(spec=LoggingWrapper)

        # Create an instance of WarmStartModule
        warm_start_module = WarmStartModule(lm_config=mock_lm_config, 
                                            runner_argument=mock_runner_argument,
                                            logging_wrapper=mock_logging_wrapper)

        # Mock the return values
        mock_conversation_history = [mock.MagicMock(spec=ConversationTurn) for _ in range(3)]
        mock_experts = ['Expert1', 'Expert2', 'Expert3']
        mock_warmstart_conversation.return_value.conversation_history = mock_conversation_history
        mock_warmstart_conversation.return_value.experts = mock_experts
        mock_report_to_conversation.return_value = [mock.MagicMock(spec=ConversationTurn) for _ in range(2)]

        # Create a mock KnowledgeBase
        mock_knowledge_base = mock.MagicMock(spec=KnowledgeBase)

        # Call the method
        result = warm_start_module.initiate_warm_start("Test Topic", mock_knowledge_base)

        # Assert the results
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], List)
        self.assertIsInstance(result[1], List)
        self.assertIsInstance(result[2], List)

        # Check if the methods were called
        mock_warmstart_conversation.assert_called_once()
        mock_generate_warmstart_outline_module.assert_called_once()
        mock_knowledge_base.insert_from_outline_string.assert_called_once()
        mock_knowledge_base.to_report.assert_called_once()
        mock_report_to_conversation.assert_called_once()

        # Check if the conversation history and experts are correctly returned
        self.assertEqual(result[0], mock_conversation_history)
        self.assertEqual(result[2], mock_experts)