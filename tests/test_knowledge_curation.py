import unittest

from knowledge_storm.interface import Retriever
from knowledge_storm.storm_wiki.modules.knowledge_curation import StormKnowledgeCurationModule
from knowledge_storm.storm_wiki.modules.persona_generator import StormPersonaGenerator
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormInformationTable
from unittest.mock import Mock, patch

class TestStormKnowledgeCurationModule(unittest.TestCase):
    @patch('knowledge_storm.storm_wiki.modules.knowledge_curation.ConvSimulator')
    def test_research_with_disabled_perspective(self, mock_conv_simulator):
        # Mock dependencies
        mock_retriever = Mock(spec=Retriever)
        mock_persona_generator = Mock(spec=StormPersonaGenerator)
        mock_lm = Mock()
        mock_callback_handler = Mock()

        # Create an instance of StormKnowledgeCurationModule
        module = StormKnowledgeCurationModule(
            retriever=mock_retriever,
            persona_generator=mock_persona_generator,
            conv_simulator_lm=mock_lm,
            question_asker_lm=mock_lm,
            max_search_queries_per_turn=3,
            search_top_k=5,
            max_conv_turn=10,
            max_thread_num=4
        )

        # Mock the ConvSimulator's return value
        mock_conv_result = Mock()
        mock_conv_result.dlg_history = []
        mock_conv_simulator.return_value.return_value = mock_conv_result

        # Call the research method with disabled perspective
        result = module.research(
            topic="Test Topic",
            ground_truth_url="http://example.com",
            callback_handler=mock_callback_handler,
            disable_perspective=True
        )

        # Assert that the result is an instance of StormInformationTable
        self.assertIsInstance(result, StormInformationTable)

        # Assert that ConvSimulator was called with an empty persona
        mock_conv_simulator.return_value.assert_called_once_with(
            topic="Test Topic",
            ground_truth_url="http://example.com",
            persona="",
            callback_handler=mock_callback_handler
        )

        # Assert that the persona generator was not called
        mock_persona_generator.generate_persona.assert_not_called()

        # Assert that the callback handler methods were called
        mock_callback_handler.on_identify_perspective_start.assert_called_once()
        mock_callback_handler.on_identify_perspective_end.assert_called_once_with(perspectives=[""])
        mock_callback_handler.on_information_gathering_start.assert_called_once()
        mock_callback_handler.on_information_gathering_end.assert_called_once()