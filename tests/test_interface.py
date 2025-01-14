import unittest

from knowledge_storm.interface import Article, ArticleSectionNode, Information

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

    def test_information_serialization(self):
        # Create a dictionary representing an Information object
        info_dict = {
            "url": "https://example.com/test",
            "description": "Test description",
            "snippets": ["Snippet A", "Snippet B"],
            "title": "Test Title",
            "meta": {"question": "Test question", "query": "Test query"},
            "citation_uuid": 12345
        }

        # Create Information object from dictionary
        info = Information.from_dict(info_dict)

        # Check if the object was created correctly
        self.assertEqual(info.url, info_dict["url"])
        self.assertEqual(info.description, info_dict["description"])
        self.assertEqual(info.snippets, info_dict["snippets"])
        self.assertEqual(info.title, info_dict["title"])
        self.assertEqual(info.meta, info_dict["meta"])
        self.assertEqual(info.citation_uuid, info_dict["citation_uuid"])

        # Convert the object back to a dictionary
        result_dict = info.to_dict()

        # Compare the original and resulting dictionaries
        self.assertEqual(info_dict, result_dict, "The original and resulting dictionaries should be identical")

        # Test with missing citation_uuid
        info_dict_no_uuid = info_dict.copy()
        del info_dict_no_uuid["citation_uuid"]
        info_no_uuid = Information.from_dict(info_dict_no_uuid)
        self.assertEqual(info_no_uuid.citation_uuid, -1, "citation_uuid should default to -1 if not provided")

    def test_meta_str_impact_on_equality_and_hashing(self):
        # Create two Information objects with the same content but different meta dictionaries
        info1 = Information(
            url="https://example.com",
            description="Test description",
            snippets=["Snippet 1", "Snippet 2"],
            title="Test Title",
            meta={"question": "Test question", "query": "Test query", "irrelevant": "This shouldn't matter"}
        )
        info2 = Information(
            url="https://example.com",
            description="Test description",
            snippets=["Snippet 1", "Snippet 2"],
            title="Test Title",
            meta={"question": "Test question", "query": "Test query", "different": "This also shouldn't matter"}
        )

        # Test equality
        self.assertEqual(info1, info2, "Information objects should be equal despite different irrelevant meta data")

        # Test hashing
        self.assertEqual(hash(info1), hash(info2), "Hash values should be equal despite different irrelevant meta data")

        # Create a third Information object with different relevant meta data
        info3 = Information(
            url="https://example.com",
            description="Test description",
            snippets=["Snippet 1", "Snippet 2"],
            title="Test Title",
            meta={"question": "Different question", "query": "Test query"}
        )

        # Test inequality
        self.assertNotEqual(info1, info3, "Information objects should not be equal due to different relevant meta data")

        # Test different hash values
        self.assertNotEqual(hash(info1), hash(info3), "Hash values should be different due to different relevant meta data")

class TestArticleSectionNode(unittest.TestCase):
    def test_add_and_remove_children(self):
        # Create parent and child nodes
        parent = ArticleSectionNode("Parent")
        child1 = ArticleSectionNode("Child 1")
        child2 = ArticleSectionNode("Child 2")

        # Add children to parent
        parent.add_child(child1)
        parent.add_child(child2, insert_to_front=True)

        # Check if children are added in the correct order
        self.assertEqual(len(parent.children), 2)
        self.assertEqual(parent.children[0], child2)
        self.assertEqual(parent.children[1], child1)

        # Remove a child
        parent.remove_child(child1)

        # Check if the child is removed
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(parent.children[0], child2)
        self.assertNotIn(child1, parent.children)

class TestArticle(unittest.TestCase):
    def test_get_outline_tree(self):
        # Create a mock article structure
        article = Article("Main Topic")

        # Add first-level sections
        intro = ArticleSectionNode("Introduction")
        methods = ArticleSectionNode("Methods")
        results = ArticleSectionNode("Results")
        article.root.add_child(intro)
        article.root.add_child(methods)
        article.root.add_child(results)

        # Add second-level sections
        background = ArticleSectionNode("Background")
        objective = ArticleSectionNode("Objective")
        intro.add_child(background)
        intro.add_child(objective)

        data_collection = ArticleSectionNode("Data Collection")
        analysis = ArticleSectionNode("Analysis")
        methods.add_child(data_collection)
        methods.add_child(analysis)

        # Generate the outline tree
        outline_tree = article.get_outline_tree()

        # Define the expected outline structure
        expected_outline = {
            'Introduction': {
                'Background': {},
                'Objective': {}
            },
            'Methods': {
                'Data Collection': {},
                'Analysis': {}
            },
            'Results': {}
        }

        # Assert that the generated outline matches the expected structure
        self.assertEqual(outline_tree, expected_outline)