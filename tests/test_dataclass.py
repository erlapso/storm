import unittest

from knowledge_storm.dataclass import KnowledgeBase, KnowledgeNode
from unittest.mock import Mock

class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        # Mock the dspy.dsp.LM dependency
        mock_lm = Mock()
        self.knowledge_base = KnowledgeBase("Test Topic", mock_lm, node_expansion_trigger_count=5)

    def test_insert_from_outline_string(self):
        outline_string = """
        # Root
        ## Topic 1
        ### Subtopic 1A
        ### Subtopic 1B
        ## Topic 2
        ### Subtopic 2A
        #### Subsubtopic 2A1
        """
        self.knowledge_base.insert_from_outline_string(outline_string)

        # Check if the root node has two children (Topic 1 and Topic 2)
        self.assertEqual(len(self.knowledge_base.root.children), 2)

        # Check if Topic 1 has two children (Subtopic 1A and Subtopic 1B)
        topic_1 = self.knowledge_base.find_node(self.knowledge_base.root, "Topic 1")
        self.assertIsNotNone(topic_1)
        self.assertEqual(len(topic_1.children), 2)

        # Check if Topic 2 has one child (Subtopic 2A)
        topic_2 = self.knowledge_base.find_node(self.knowledge_base.root, "Topic 2")
        self.assertIsNotNone(topic_2)
        self.assertEqual(len(topic_2.children), 1)

        # Check if Subtopic 2A has one child (Subsubtopic 2A1)
        subtopic_2a = self.knowledge_base.find_node(topic_2, "Subtopic 2A")
        self.assertIsNotNone(subtopic_2a)
        self.assertEqual(len(subtopic_2a.children), 1)

        # Check if the deepest node (Subsubtopic 2A1) exists and has no children
        subsubtopic_2a1 = self.knowledge_base.find_node(subtopic_2a, "Subsubtopic 2A1")
        self.assertIsNotNone(subsubtopic_2a1)
        self.assertEqual(len(subsubtopic_2a1.children), 0)