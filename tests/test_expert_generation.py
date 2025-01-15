import pytest

from knowledge_storm.collaborative_storm.modules.expert_generation import GenerateExpertModule
from unittest.mock import Mock

class TestGenerateExpertModule:
    def test_generate_expert_with_focus(self):
        # Mock the LM engine
        mock_engine = Mock()

        # Create an instance of GenerateExpertModule with the mock engine
        expert_module = GenerateExpertModule(engine=mock_engine)

        # Mock the generate_expert_w_focus method to return a predefined output
        expert_module.generate_expert_w_focus = Mock(return_value=Mock(experts="""
        1. University Professor: Expert in environmental science focusing on climate change impacts
        2. Industry Representative: Spokesperson for a major oil company discussing adaptation strategies
        3. Environmental Activist: Leader of a grassroots organization advocating for immediate action
        """))

        # Test parameters
        topic = "Climate Change"
        num_experts = 3
        background_info = "Recent studies show accelerating global temperature rise."
        focus = "Adaptation strategies for coastal cities"

        # Call the forward method
        result = expert_module.forward(topic, num_experts, background_info, focus)

        # Assert that the output is as expected
        assert len(result.experts) == 3
        assert "University Professor" in result.experts[0]
        assert "Industry Representative" in result.experts[1]
        assert "Environmental Activist" in result.experts[2]

        # Verify that generate_expert_w_focus was called with the correct parameters
        expert_module.generate_expert_w_focus.assert_called_once_with(
            topic=topic,
            background_info=background_info,
            focus=focus,
            topN=num_experts
        )