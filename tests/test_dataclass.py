import unittest

from knowledge_storm.dataclass import KnowledgeBase
from unittest.mock import MagicMock

class TestKnowledgeBase(unittest.TestCase):
    def test_insert_from_outline_string(self):
        # Mock the knowledge_base_lm
        mock_lm = MagicMock()

        # Create a KnowledgeBase instance
        kb = KnowledgeBase(topic="Test Topic", knowledge_base_lm=mock_lm, node_expansion_trigger_count=5)

        # Define an outline string
        outline_string = """
        # Root
        ## Topic 1
        ### Subtopic 1A
        ### Subtopic 1B
        ## Topic 2
        ### Subtopic 2A
        #### Subsubtopic 2A1
        """

        # Insert nodes from the outline string
        kb.insert_from_outline_string(outline_string)

        # Check if the nodes are correctly inserted
        root = kb.root
        self.assertEqual(len(root.children), 2)

        topic1 = root.children[0]
        self.assertEqual(topic1.name, "Topic 1")
        self.assertEqual(len(topic1.children), 2)
        self.assertEqual(topic1.children[0].name, "Subtopic 1A")
        self.assertEqual(topic1.children[1].name, "Subtopic 1B")

        topic2 = root.children[1]
        self.assertEqual(topic2.name, "Topic 2")
        self.assertEqual(len(topic2.children), 1)

        subtopic2A = topic2.children[0]
        self.assertEqual(subtopic2A.name, "Subtopic 2A")
        self.assertEqual(len(subtopic2A.children), 1)
        self.assertEqual(subtopic2A.children[0].name, "Subsubtopic 2A1")