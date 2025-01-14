import unittest

from knowledge_storm.storm_wiki.modules.persona_generator import StormPersonaGenerator
from unittest.mock import Mock, patch

class TestStormPersonaGenerator(unittest.TestCase):
    @patch('knowledge_storm.storm_wiki.modules.persona_generator.CreateWriterWithPersona')
    def test_generate_persona_with_specific_topic(self, mock_create_writer):
        # Arrange
        mock_engine = Mock()
        mock_create_writer.return_value.return_value = Mock(
            personas=['Persona 1: Description 1', 'Persona 2: Description 2', 'Persona 3: Description 3']
        )
        generator = StormPersonaGenerator(mock_engine)
        topic = "Artificial Intelligence"
        max_num_persona = 2

        # Act
        result = generator.generate_persona(topic, max_num_persona)

        # Assert
        self.assertEqual(len(result), 3)  # 2 generated + 1 default
        self.assertTrue(result[0].startswith("Basic fact writer:"))
        self.assertEqual(result[1], "Persona 1: Description 1")
        self.assertEqual(result[2], "Persona 2: Description 2")
        mock_create_writer.return_value.assert_called_once_with(topic=topic)