import dspy
import unittest

from knowledge_storm.collaborative_storm.modules.information_insertion_module import InsertInformationModule
from knowledge_storm.dataclass import KnowledgeBase, KnowledgeNode
from unittest.mock import MagicMock, patch

class TestInsertInformationModule(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.module = InsertInformationModule(self.engine)

    def test_layer_by_layer_navigation_placement(self):
        # Create a simple knowledge base structure
        root = KnowledgeNode(name="Root")
        child1 = KnowledgeNode(name="Child1")
        child2 = KnowledgeNode(name="Child2")
        root.add_child(child1)
        root.add_child(child2)
        kb = KnowledgeBase(root=root)

        # Mock the _get_navigation_choice method to simulate navigation
        self.module._get_navigation_choice = MagicMock(side_effect=[
            ("step", "Child1"),
            ("insert", "")
        ])

        # Call the method
        result = self.module.layer_by_layer_navigation_placement(
            knowledge_base=kb,
            question="Test question",
            query="Test query"
        )

        # Assert the result
        self.assertIsInstance(result, dspy.Prediction)
        self.assertEqual(result.information_placement, "Root -> Child1")
        self.assertEqual(result.note, "None")

        # Verify that _get_navigation_choice was called twice
        self.assertEqual(self.module._get_navigation_choice.call_count, 2)