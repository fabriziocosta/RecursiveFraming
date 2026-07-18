import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd

from graphicalizer import NetworkXGraphStore, process_abstract_folder, process_corpus_articles


class FakeGraphicalizer:
    def run(self, text):
        if not text.strip():
            raise ValueError("empty abstract")
        graph = nx.MultiDiGraph()
        graph.add_node("e1", node_context=text.strip())
        return SimpleNamespace(graph=graph)


class BatchTests(unittest.TestCase):
    def test_processes_all_matching_files_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "abstracts"
            folder.mkdir()
            (folder / "b.txt").write_text("second abstract", encoding="utf-8")
            (folder / "a.txt").write_text("first abstract", encoding="utf-8")
            (folder / "empty.txt").write_text("", encoding="utf-8")
            store = NetworkXGraphStore(Path(directory) / "graphs")

            result = process_abstract_folder(folder, FakeGraphicalizer(), store)

            self.assertEqual(result.discovered, 3)
            self.assertEqual(result.processed, 2)
            self.assertEqual(result.failed, 1)
            self.assertEqual(
                [path.name for path in result.saved_paths],
                ["abstract_0000_a.gpickle", "abstract_0001_b.gpickle"],
            )
            self.assertTrue(result.manifest_path.exists())
            self.assertEqual(len(store.list()), 2)
            self.assertEqual(
                store.load("abstract_0000_a").graph["source_path"],
                str(folder / "a.txt"),
            )

    def test_can_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "abstracts"
            folder.mkdir()
            (folder / "a.txt").write_text("", encoding="utf-8")
            store = NetworkXGraphStore(Path(directory) / "graphs")

            with self.assertRaises(ValueError):
                process_abstract_folder(
                    folder,
                    FakeGraphicalizer(),
                    store,
                    continue_on_error=False,
                )

    def test_processes_filtered_corpus_rows_and_preserves_provenance(self):
        frame = pd.DataFrame(
            [
                {
                    "pathogen": "Nipah virus",
                    "pmid": "101",
                    "title": "First study",
                    "abstract": "First abstract",
                    "publication_date": "2020",
                },
                {
                    "pathogen": "Nipah virus",
                    "pmid": "102",
                    "title": "Second study",
                    "abstract": "Second abstract",
                    "publication_date": "2021",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            store = NetworkXGraphStore(Path(directory) / "graphs")
            result = process_corpus_articles(
                frame,
                FakeGraphicalizer(),
                store,
                corpus_path=Path(directory) / "corpus_articles.parquet",
                verbose=False,
            )

            self.assertEqual(result.discovered, 2)
            self.assertEqual(result.processed, 2)
            graph = store.load("pubmed_0000_Nipah_virus_101")
            self.assertEqual(graph.graph["pathogen"], "Nipah virus")
            self.assertEqual(graph.graph["pmid"], "101")
            self.assertEqual(graph.graph["publication_date"], "2020")


if __name__ == "__main__":
    unittest.main()
