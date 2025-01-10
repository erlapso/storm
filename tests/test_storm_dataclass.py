import re
import unittest

from knowledge_storm.storm_wiki.modules.storm_dataclass import ArticleSectionNode, Information, StormArticle

class TestStormArticle(unittest.TestCase):
    def test_from_outline_str(self):
        topic = "Test Topic"
        outline_str = """
        # Test Topic
        ## Section 1
        ### Subsection 1.1
        ## Section 2
        ### Subsection 2.1
        ### Subsection 2.2
        """

        article = StormArticle.from_outline_str(topic, outline_str)

        # Check if the article has the correct topic
        self.assertEqual(article.topic_name, topic)

        # Check if the root has two children (Section 1 and Section 2)
        self.assertEqual(len(article.root.children), 2)

        # Check Section 1
        section1 = article.root.children[0]
        self.assertEqual(section1.section_name, "Section 1")
        self.assertEqual(len(section1.children), 1)
        self.assertEqual(section1.children[0].section_name, "Subsection 1.1")

        # Check Section 2
        section2 = article.root.children[1]
        self.assertEqual(section2.section_name, "Section 2")
        self.assertEqual(len(section2.children), 2)
        self.assertEqual(section2.children[0].section_name, "Subsection 2.1")
        self.assertEqual(section2.children[1].section_name, "Subsection 2.2")

    def test_update_section(self):
        # Create a StormArticle instance
        article = StormArticle("Test Topic")

        # Create some test information
        info1 = Information(url="https://example.com/1", snippets=["Snippet 1"])
        info2 = Information(url="https://example.com/2", snippets=["Snippet 2"])

        # Test content with citations
        content = "This is a test section with citations [1] and [2]."

        # Update the root section
        article.update_section(content, [info1, info2])

        # Check if the content was updated correctly
        root_content = article.root.children[0].content
        self.assertEqual(root_content, content)

        # Check if references were added correctly
        self.assertEqual(len(article.reference["url_to_info"]), 2)
        self.assertIn("https://example.com/1", article.reference["url_to_info"])
        self.assertIn("https://example.com/2", article.reference["url_to_info"])

        # Check if citation indices are correct
        citation_indices = re.findall(r'\[(\d+)\]', root_content)
        self.assertEqual(citation_indices, ['1', '2'])

        # Update the section with new content and references
        new_content = "Updated content with citations [1], [2], and [3]."
        info3 = Information(url="https://example.com/3", snippets=["Snippet 3"])
        article.update_section(new_content, [info1, info2, info3])

        # Check if the content was updated
        updated_content = article.root.children[0].content
        self.assertEqual(updated_content, new_content)

        # Check if the new reference was added
        self.assertEqual(len(article.reference["url_to_info"]), 3)
        self.assertIn("https://example.com/3", article.reference["url_to_info"])

        # Check if citation indices are still correct
        updated_citation_indices = re.findall(r'\[(\d+)\]', updated_content)
        self.assertEqual(updated_citation_indices, ['1', '2', '3'])

    def test_get_outline_as_list(self):
        # Create a StormArticle instance with a predefined structure
        article = StormArticle("Test Topic")
        article.update_section("Root content", [])
        article.update_section("Section 1 content", [], "Test Topic")
        article.update_section("Subsection 1.1 content", [], "Section 1")
        article.update_section("Section 2 content", [], "Test Topic")
        article.update_section("Subsection 2.1 content", [], "Section 2")
        article.update_section("Subsection 2.2 content", [], "Section 2")

        # Test get_outline_as_list with default parameters
        outline = article.get_outline_as_list()
        expected_outline = ["Test Topic", "Section 1", "Subsection 1.1", "Section 2", "Subsection 2.1", "Subsection 2.2"]
        self.assertEqual(outline, expected_outline)

        # Test get_outline_as_list with add_hashtags=True
        outline_with_hashtags = article.get_outline_as_list(add_hashtags=True)
        expected_outline_with_hashtags = ["# Test Topic", "## Section 1", "### Subsection 1.1", "## Section 2", "### Subsection 2.1", "### Subsection 2.2"]
        self.assertEqual(outline_with_hashtags, expected_outline_with_hashtags)

        # Test get_outline_as_list with a specific root_section_name
        section_2_outline = article.get_outline_as_list(root_section_name="Section 2")
        expected_section_2_outline = ["Section 2", "Subsection 2.1", "Subsection 2.2"]
        self.assertEqual(section_2_outline, expected_section_2_outline)

        # Test get_outline_as_list with include_root=False
        outline_without_root = article.get_outline_as_list(include_root=False)
        expected_outline_without_root = ["Section 1", "Subsection 1.1", "Section 2", "Subsection 2.1", "Subsection 2.2"]
        self.assertEqual(outline_without_root, expected_outline_without_root)

if __name__ == '__main__':
    unittest.main()

    def test_reorder_reference_index(self):
        # Create a StormArticle instance
        article = StormArticle("Test Topic")

        # Create some test information
        info1 = Information(url="https://example.com/1", snippets=["Snippet 1"])
        info2 = Information(url="https://example.com/2", snippets=["Snippet 2"])
        info3 = Information(url="https://example.com/3", snippets=["Snippet 3"])

        # Add sections with out-of-order references
        article.update_section("Section 1 with reference [2].", [info2], "Test Topic")
        article.update_section("Section 2 with reference [1] and [3].", [info1, info3], "Test Topic")

        # Initial state of references
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/1"], 1)
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/2"], 2)
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/3"], 3)

        # Call the reorder_reference_index method
        article.reorder_reference_index()

        # Check if the reference indices have been reordered
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/2"], 1)
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/1"], 2)
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/3"], 3)

        # Check if the content has been updated with new reference numbers
        section1 = article.find_section(article.root, "Section 1")
        section2 = article.find_section(article.root, "Section 2")
        self.assertEqual(section1.content, "Section 1 with reference [1].")
        self.assertEqual(section2.content, "Section 2 with reference [2] and [3].")

    def test_get_outline_tree(self):
        # Create a StormArticle instance with a predefined structure
        article = StormArticle("Test Topic")
        article.update_section("Root content", [])
        article.update_section("Section 1 content", [], "Test Topic")
        article.update_section("Subsection 1.1 content", [], "Section 1")
        article.update_section("Section 2 content", [], "Test Topic")
        article.update_section("Subsection 2.1 content", [], "Section 2")
        article.update_section("Subsection 2.2 content", [], "Section 2")

        # Get the outline tree
        outline_tree = article.get_outline_tree()

        # Expected outline tree structure
        expected_tree = {
            "Section 1": {
                "Subsection 1.1": {}
            },
            "Section 2": {
                "Subsection 2.1": {},
                "Subsection 2.2": {}
            }
        }

        # Assert that the outline tree matches the expected structure
        self.assertEqual(outline_tree, expected_tree)

        # Test with an empty article
        empty_article = StormArticle("Empty Topic")
        empty_tree = empty_article.get_outline_tree()
        self.assertEqual(empty_tree, {})

    def test_from_string(self):
        # Mock article text
        article_text = """# Test Topic

## Section 1
This is content for Section 1 with a reference [1].

## Section 2
This is content for Section 2 with references [1] and [2].

### Subsection 2.1
This is content for Subsection 2.1."""

        # Mock references
        references = {
            "url_to_unified_index": {
                "https://example.com/1": 1,
                "https://example.com/2": 2
            },
            "url_to_info": {
                "https://example.com/1": {
                    "url": "https://example.com/1",
                    "snippets": ["Snippet 1"]
                },
                "https://example.com/2": {
                    "url": "https://example.com/2",
                    "snippets": ["Snippet 2"]
                }
            }
        }

        # Create StormArticle from string
        article = StormArticle.from_string("Test Topic", article_text, references)

        # Verify the article structure
        self.assertEqual(article.topic_name, "Test Topic")
        self.assertEqual(len(article.root.children), 2)
        self.assertEqual(article.root.children[0].section_name, "Section 1")
        self.assertEqual(article.root.children[1].section_name, "Section 2")
        self.assertEqual(len(article.root.children[1].children), 1)
        self.assertEqual(article.root.children[1].children[0].section_name, "Subsection 2.1")

        # Verify the content
        self.assertIn("This is content for Section 1", article.root.children[0].content)
        self.assertIn("This is content for Section 2", article.root.children[1].content)
        self.assertIn("This is content for Subsection 2.1", article.root.children[1].children[0].content)

        # Verify the references
        self.assertEqual(len(article.reference["url_to_info"]), 2)
        self.assertIsInstance(article.reference["url_to_info"]["https://example.com/1"], Information)
        self.assertEqual(article.reference["url_to_info"]["https://example.com/1"].url, "https://example.com/1")
        self.assertEqual(article.reference["url_to_info"]["https://example.com/1"].snippets, ["Snippet 1"])
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/1"], 1)
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/2"], 2)

    def test_post_processing(self):
        # Create a StormArticle instance
        article = StormArticle("Test Topic")

        # Create some test information
        info1 = Information(url="https://example.com/1", snippets=["Snippet 1"])
        info2 = Information(url="https://example.com/2", snippets=["Snippet 2"])

        # Add sections with out-of-order references and an empty section
        article.update_section("Section 1 with reference [2].", [info2], "Test Topic")
        article.update_section("", [], "Test Topic")  # Empty section
        article.update_section("Section 2 with reference [1].", [info1], "Test Topic")

        # Call the post_processing method
        article.post_processing()

        # Check if the empty section was removed
        self.assertEqual(len(article.root.children), 2)
        self.assertEqual(article.root.children[0].section_name, "Section 1")
        self.assertEqual(article.root.children[1].section_name, "Section 2")

        # Check if the reference indices have been reordered
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/2"], 1)
        self.assertEqual(article.reference["url_to_unified_index"]["https://example.com/1"], 2)

        # Check if the content has been updated with new reference numbers
        section1 = article.find_section(article.root, "Section 1")
        section2 = article.find_section(article.root, "Section 2")
        self.assertEqual(section1.content, "Section 1 with reference [1].")
        self.assertEqual(section2.content, "Section 2 with reference [2].")

    def test_get_first_level_section_names(self):
        # Create a StormArticle instance with a predefined structure
        article = StormArticle("Test Topic")
        article.update_section("Root content", [])
        article.update_section("Section 1 content", [], "Test Topic")
        article.update_section("Subsection 1.1 content", [], "Section 1")
        article.update_section("Section 2 content", [], "Test Topic")
        article.update_section("Subsection 2.1 content", [], "Section 2")

        # Get the first level section names
        first_level_sections = article.get_first_level_section_names()

        # Assert that the returned list contains the expected section names
        expected_sections = ["Section 1", "Section 2"]
        self.assertEqual(first_level_sections, expected_sections)

        # Test with an empty article
        empty_article = StormArticle("Empty Topic")
        empty_sections = empty_article.get_first_level_section_names()
        self.assertEqual(empty_sections, [])