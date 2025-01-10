import dspy
import numpy as np
import unittest

from knowledge_storm.collaborative_storm.modules.information_insertion_module import ExpandNodeModule, InsertInformationModule
from knowledge_storm.dataclass import Information, KnowledgeBase, KnowledgeNode
from knowledge_storm.interface import Information
from unittest.mock import MagicMock, Mock, patch

class TestInsertInformationModule(unittest.TestCase):

    def test_choose_candidate_from_embedding_ranking(self):
        # Mock the engine
        mock_engine = Mock(spec=dspy.dsp.LM)

        # Create an instance of InsertInformationModule
        insert_module = InsertInformationModule(engine=mock_engine)

        # Mock input data
        question = "What is the capital of France?"
        query = "capital of France"
        encoded_outlines = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        outlines = ["Geography -> Europe -> France", "History -> European Countries", "Culture -> Western Europe"]

        # Mock the candidate_choosing method
        mock_prediction = Mock(spec=dspy.Prediction)
        mock_prediction.decision = "Best placement: [1]"
        insert_module.candidate_choosing = Mock(return_value=mock_prediction)

        # Mock get_text_embeddings function
        with patch('knowledge_storm.collaborative_storm.modules.information_insertion_module.get_text_embeddings') as mock_get_embeddings:
            mock_get_embeddings.return_value = (np.array([0.1, 0.2, 0.3]), 10)  # Mock embedding and token usage

            # Call the method
            result = insert_module.choose_candidate_from_embedding_ranking(
                question, query, encoded_outlines, outlines
            )

        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dspy.Prediction)
        self.assertEqual(result.information_placement, "Geography -> Europe -> France")
        self.assertTrue(result.note.startswith("Choosing from:"))

        # Verify that the candidate_choosing method was called with correct arguments
        insert_module.candidate_choosing.assert_called_once()
        call_args = insert_module.candidate_choosing.call_args[1]
        self.assertIn("Question: What is the capital of France?", call_args['intent'])
        self.assertIn("Query: capital of France", call_args['intent'])
        self.assertIn("1: Geography -> Europe -> France", call_args['choices'])

    def test_layer_by_layer_navigation_placement(self):
        # Mock the engine
        mock_engine = MagicMock(spec=dspy.dsp.LM)

        # Create an instance of InsertInformationModule
        insert_module = InsertInformationModule(engine=mock_engine)

        # Create a mock KnowledgeBase with a simple tree structure
        mock_kb = MagicMock(spec=KnowledgeBase)
        root_node = KnowledgeNode(name="Root")
        child_node = KnowledgeNode(name="Child1")
        grandchild_node = KnowledgeNode(name="GrandChild1")
        root_node.add_child(child_node)
        child_node.add_child(grandchild_node)
        mock_kb.root = root_node

        # Mock the _get_navigation_choice method to simulate navigation
        with patch.object(insert_module, '_get_navigation_choice') as mock_get_choice:
            mock_get_choice.side_effect = [
                ("step", "Child1"),
                ("step", "GrandChild1"),
                ("insert", "")
            ]

            # Call the method
            result = insert_module.layer_by_layer_navigation_placement(
                knowledge_base=mock_kb,
                question="Test question",
                query="Test query"
            )

        # Assertions
        self.assertIsInstance(result, dspy.Prediction)
        self.assertEqual(result.information_placement, "Root -> Child1 -> GrandChild1")
        self.assertEqual(result.note, "None")

        # Verify that _get_navigation_choice was called the expected number of times
        self.assertEqual(mock_get_choice.call_count, 3)

        # Verify the arguments passed to _get_navigation_choice
        calls = mock_get_choice.call_args_list
        self.assertEqual(calls[0][1]['knowledge_node'], root_node)
        self.assertEqual(calls[1][1]['knowledge_node'], child_node)
        self.assertEqual(calls[2][1]['knowledge_node'], grandchild_node)

    def test_forward_multiple_information(self):
        # Mock the engine
        mock_engine = MagicMock(spec=dspy.dsp.LM)

        # Create an instance of InsertInformationModule
        insert_module = InsertInformationModule(engine=mock_engine)

        # Create a mock KnowledgeBase
        mock_kb = MagicMock(spec=KnowledgeBase)
        mock_kb.get_knowledge_base_structure_embedding.return_value = (None, [])

        # Create mock Information objects
        info1 = Information(content="Info 1", meta={"question": "Q1", "query": "Query1"})
        info2 = Information(content="Info 2", meta={"question": "Q2", "query": "Query2"})

        # Mock the choose_candidate_from_embedding_ranking method
        with patch.object(insert_module, 'choose_candidate_from_embedding_ranking', return_value=None):
            # Mock the layer_by_layer_navigation_placement method
            with patch.object(insert_module, 'layer_by_layer_navigation_placement') as mock_navigate:
                mock_navigate.side_effect = [
                    dspy.Prediction(information_placement="Root -> Node1", note="None"),
                    dspy.Prediction(information_placement="Root -> Node2", note="None")
                ]

                # Call the forward method
                result = insert_module.forward(mock_kb, [info1, info2])

        # Assertions
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], info1)
        self.assertEqual(result[0][1].information_placement, "Root -> Node1")
        self.assertEqual(result[1][0], info2)
        self.assertEqual(result[1][1].information_placement, "Root -> Node2")

        # Verify that insert_information was called twice on the mock_kb
        self.assertEqual(mock_kb.insert_information.call_count, 2)

        # Verify the arguments of insert_information calls
        calls = mock_kb.insert_information.call_args_list
        self.assertEqual(calls[0][1]['path'], "Root -> Node1")
        self.assertEqual(calls[0][1]['information'], info1)
        self.assertEqual(calls[1][1]['path'], "Root -> Node2")
        self.assertEqual(calls[1][1]['information'], info2)

class TestExpandNodeModule(unittest.TestCase):
    def test_forward_expand_node(self):
        # Mock the engine
        mock_engine = MagicMock(spec=dspy.dsp.LM)

        # Mock the information_insert_module
        mock_insert_module = MagicMock()

        # Create an instance of ExpandNodeModule
        expand_module = ExpandNodeModule(engine=mock_engine, information_insert_module=mock_insert_module, node_expansion_trigger_count=2)

        # Create a mock KnowledgeBase with a simple tree structure
        mock_kb = MagicMock(spec=KnowledgeBase)
        root_node = KnowledgeNode(name="Root")
        child_node = KnowledgeNode(name="Child1")
        root_node.add_child(child_node)
        mock_kb.root = root_node

        # Add content to the child node to trigger expansion
        child_node.content = set(['info1', 'info2'])

        # Mock the _get_expand_subnode_names method to return some subsection names
        with patch.object(expand_module, '_get_expand_subnode_names', return_value=['Subsection1', 'Subsection2']):
            # Mock the info_uuid_to_info_dict
            mock_kb.info_uuid_to_info_dict = {
                'info1': MagicMock(),
                'info2': MagicMock()
            }

            # Call the forward method
            expand_module.forward(knowledge_base=mock_kb)

        # Assertions
        self.assertEqual(len(child_node.children), 2)
        self.assertEqual(child_node.children[0].name, 'Subsection1')
        self.assertEqual(child_node.children[1].name, 'Subsection2')

        # Verify that insert_information was called on the mock_insert_module
        mock_insert_module.assert_called_once()
        call_args = mock_insert_module.call_args[1]
        self.assertEqual(call_args['knowledge_base'], mock_kb)
        self.assertEqual(len(call_args['information']), 2)
        self.assertEqual(call_args['allow_create_new_node'], False)
        self.assertEqual(call_args['insert_root'], child_node)

    def test_get_navigation_choice(self):
        # Mock the engine
        mock_engine = MagicMock()

        # Create an instance of InsertInformationModule
        insert_module = InsertInformationModule(engine=mock_engine)

        # Create a mock KnowledgeNode
        mock_node = KnowledgeNode(name="Root")
        mock_node.add_child(KnowledgeNode(name="Child1"))
        mock_node.add_child(KnowledgeNode(name="Child2"))

        # Test case 1: Insert action
        with patch.object(insert_module.insert_info, '__call__') as mock_insert_info:
            mock_insert_info.return_value.choice = "Choice:\ninsert"
            action_type, node_name = insert_module._get_navigation_choice(
                knowledge_node=mock_node,
                question="Test question",
                query="Test query"
            )
            self.assertEqual(action_type, "insert")
            self.assertEqual(node_name, "")

        # Test case 2: Step action
        with patch.object(insert_module.insert_info, '__call__') as mock_insert_info:
            mock_insert_info.return_value.choice = "Choice:\nstep: Child1"
            action_type, node_name = insert_module._get_navigation_choice(
                knowledge_node=mock_node,
                question="Test question",
                query="Test query"
            )
            self.assertEqual(action_type, "step")
            self.assertEqual(node_name, "Child1")

        # Test case 3: Create action
        with patch.object(insert_module.insert_info, '__call__') as mock_insert_info:
            mock_insert_info.return_value.choice = "Choice:\ncreate: NewChild"
            action_type, node_name = insert_module._get_navigation_choice(
                knowledge_node=mock_node,
                question="Test question",
                query="Test query"
            )
            self.assertEqual(action_type, "create")
            self.assertEqual(node_name, "NewChild")

        # Test case 4: Invalid action
        with patch.object(insert_module.insert_info, '__call__') as mock_insert_info:
            mock_insert_info.return_value.choice = "Choice:\ninvalid_action"
            with self.assertRaises(Exception):
                insert_module._get_navigation_choice(
                    knowledge_node=mock_node,
                    question="Test question",
                    query="Test query"
                )

def test_info_list_to_intent_mapping(self):
    # Mock the engine
    mock_engine = MagicMock(spec=dspy.dsp.LM)

    # Create an instance of InsertInformationModule
    insert_module = InsertInformationModule(engine=mock_engine)

    # Create a list of Information objects
    info_list = [
        Information(content="Info 1", meta={"question": "Q1", "query": "Query1"}),
        Information(content="Info 2", meta={"question": "Q2", "query": "Query2"}),
        Information(content="Info 3", meta={"question": "Q1", "query": "Query1"}),  # Duplicate intent
        Information(content="Info 4", meta={"question": "Q3", "query": ""}),  # Empty query
        Information(content="Info 5", meta={"question": "", "query": "Query4"}),  # Empty question
    ]

    # Call the method
    result = insert_module._info_list_to_intent_mapping(info_list)

    # Assertions
    self.assertIsInstance(result, dict)
    self.assertEqual(len(result), 4)  # 4 unique intents

    # Check specific intents
    self.assertIn(("Q1", "Query1"), result)
    self.assertIn(("Q2", "Query2"), result)
    self.assertIn(("Q3", ""), result)
    self.assertIn(("", "Query4"), result)

    # Check that all values are None
    self.assertTrue(all(value is None for value in result.values()))