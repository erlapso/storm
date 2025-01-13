import unittest

from knowledge_storm.interface import Information

class TestInformation(unittest.TestCase):
    def test_information_equality_and_hashing(self):
        # Create two Information objects with the same content
        info1 = Information(
            url="https://example.com",
            description="Test description",
            snippets=["Snippet 1", "Snippet 2"],
            title="Test Title",
            meta={"question": "Test question", "query": "Test query"}
        )
        info2 = Information(
            url="https://example.com",
            description="Test description",
            snippets=["Snippet 2", "Snippet 1"],  # Order shouldn't matter
            title="Test Title",
            meta={"query": "Test query", "question": "Test question"}  # Order shouldn't matter
        )

        # Test equality
        self.assertEqual(info1, info2, "Information objects with same content should be equal")

        # Test hashing
        self.assertEqual(hash(info1), hash(info2), "Hash values should be equal for equal Information objects")

        # Create a different Information object
        info3 = Information(
            url="https://example.com/different",
            description="Different description",
            snippets=["Snippet 1", "Snippet 2"],
            title="Different Title",
            meta={"question": "Different question", "query": "Different query"}
        )

        # Test inequality
        self.assertNotEqual(info1, info3, "Information objects with different content should not be equal")

        # Test different hash values
        self.assertNotEqual(hash(info1), hash(info3), "Hash values should be different for unequal Information objects")