import dspy
import unittest

from knowledge_storm.collaborative_storm.modules.information_insertion_module import InsertInformationModule
from knowledge_storm.dataclass import Information, KnowledgeBase
from unittest.mock import Mock

class TestInsertInformationModule(unittest.TestCase):
    def setUp(self):
        self.mock_engine = Mock(spec=dspy.dsp.LM)
        self.insert_module = InsertInformationModule(self.mock_engine)
        self.knowledge_base = KnowledgeBase()

    def test_insert_single_information(self):
        # Create a mock Information object
        mock_info = Information(content="Test content", meta={"question": "Test question", "query": "Test query"})

        # Mock the layer_by_layer_navigation_placement method to return a specific placement
        self.insert_module.layer_by_layer_navigation_placement = Mock(
            return_value=dspy.Prediction(information_placement="root -> child1", note="None")
        )

        # Call the forward method
        result = self.insert_module(
            knowledge_base=self.knowledge_base,
            information=mock_info,
            allow_create_new_node=False
        )

        # Assert that the result is correct
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], mock_info)
        self.assertEqual(result[0][1].information_placement, "root -> child1")

        # Assert that the information was inserted into the knowledge base
        self.assertEqual(len(self.knowledge_base.info_uuid_to_info_dict), 1)
        inserted_info = list(self.knowledge_base.info_uuid_to_info_dict.values())[0]
        self.assertEqual(inserted_info.content, "Test content")
        self.assertEqual(inserted_info.meta["question"], "Test question")
        self.assertEqual(inserted_info.meta["query"], "Test query")