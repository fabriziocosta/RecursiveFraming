import tempfile
import unittest
from pathlib import Path

from graphicalizer import PubMedArticle, PubMedClient, PubMedSearchPage


ARTICLE_XML = """<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123456</PMID>
      <Article>
        <ArticleTitle>Nipah virus study</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">A background statement.</AbstractText>
          <AbstractText>Findings are reported.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
          <Author><CollectiveName>Study Group</CollectiveName></Author>
        </AuthorList>
        <Journal>
          <Title>Example Journal</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/example</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


class PubMedTests(unittest.TestCase):
    def setUp(self):
        self.client = PubMedClient(email="researcher@example.org")

    def test_keyword_query_is_explicit(self):
        query = self.client.keyword_query(["Nipah virus", "zoonotic transmission"])
        self.assertEqual(
            query,
            '"Nipah virus"[Title/Abstract] AND "zoonotic transmission"[Title/Abstract]',
        )

    def test_article_xml_is_parsed(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(ARTICLE_XML)
        article = self.client._parse_article(root.find(".//PubmedArticle"))

        self.assertEqual(article.pmid, "123456")
        self.assertIn("BACKGROUND: A background statement.", article.abstract)
        self.assertIn("Findings are reported.", article.abstract)
        self.assertEqual(article.doi, "10.1000/example")
        self.assertEqual(article.authors, ("Jane Doe", "Study Group"))

    def test_articles_are_saved_with_manifest(self):
        article = PubMedArticle(
            pmid="123456",
            title="Nipah virus study",
            abstract="A short abstract.",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "abstracts"
            manifest = output / "manifest.json"
            paths = self.client.save_articles(
                [article],
                output,
                manifest_path=manifest,
            )

            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())
            self.assertIn("A short abstract.", paths[0].read_text())
            self.assertTrue(manifest.exists())

    def test_search_all_retrieves_pages(self):
        calls = []

        def request(endpoint, params):
            calls.append((endpoint, dict(params)))
            start = int(params.get("retstart", 0))
            ids = ["101", "102"] if start == 0 else ["103"]
            return (
                '{"esearchresult":{"count":"3","idlist":['
                + ",".join(f'"{item}"' for item in ids)
                + "]}}"
            ).encode()

        self.client._request = request
        total, ids = self.client.search_all("virus[Title/Abstract]", page_size=2)

        self.assertEqual(total, 3)
        self.assertEqual(ids, ["101", "102", "103"])
        self.assertEqual([call[1].get("retstart", 0) for call in calls], [0, 2])
        self.assertIsInstance(self.client.search_page("virus[Title/Abstract]"), PubMedSearchPage)


if __name__ == "__main__":
    unittest.main()
