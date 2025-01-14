import dspy
import unittest

from knowledge_storm.collaborative_storm.modules.expert_generation import GenerateExpertModule
from unittest.mock import Mock, patch

class TestGenerateExpertModule(unittest.TestCase):
    @patch('dspy.settings.context')
    def test_generate_expert_with_focus(self, mock_context):
        # Mock the dspy engine
        mock_engine = Mock(spec=dspy.dsp.LM)

        # Create an instance of GenerateExpertModule
        expert_module = GenerateExpertModule(engine=mock_engine)

        # Mock the generate_expert_w_focus method
        expert_module.generate_expert_w_focus = Mock()
        expert_module.generate_expert_w_focus.return_value = Mock(
            experts="1. Expert A: Description A\n2. Expert B: Description B"
        )

        # Test parameters
        topic = "AI Ethics"
        num_experts = 2
        background_info = "AI ethics is a growing concern in the tech industry."
        focus = "Bias in AI algorithms"

        # Call the forward method
        result = expert_module.forward(topic, num_experts, background_info, focus)

        # Assert that the correct method was called
        expert_module.generate_expert_w_focus.assert_called_once()

        # Assert that the result is as expected
        self.assertEqual(len(result.experts), 2)
        self.assertEqual(result.experts[0], "Expert A: Description A")
        self.assertEqual(result.experts[1], "Expert B: Description B")

        # Assert that the context manager was used
        mock_context.assert_called_once_with(lm=mock_engine, show_guidelines=False)