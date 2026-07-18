import tempfile
import unittest
from pathlib import Path

import networkx as nx

from graphicalizer import NetworkXGraphStore, NodeContextEmbedder, NodeEmbeddingConfig


class FakeEmbeddingModel:
    model_name = "fake-embeddings"

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[3.0, 4.0] for _ in texts]


class EmbeddingTests(unittest.TestCase):
    def test_node_context_summaries_are_embedded_and_normalized(self):
        graph = nx.MultiDiGraph()
        graph.add_node("e1")
        graph.add_node("e2")
        contexts = (
            type("Context", (), {"entity_id": "e1", "summary": "first summary"})(),
            type("Context", (), {"entity_id": "e2", "summary": "second summary"})(),
        )
        model = FakeEmbeddingModel()

        NodeContextEmbedder(model).enrich_graph(graph, contexts)

        self.assertEqual(model.calls, [["first summary", "second summary"]])
        self.assertEqual(graph.nodes["e1"]["node_context_embedding"], [0.6, 0.8])
        self.assertEqual(graph.nodes["e2"]["node_context_embedding_model"], "fake-embeddings")
        self.assertEqual(graph.graph["node_context_embeddings"]["dimension"], 2)

    def test_graph_store_saves_one_file_per_graph_and_loads_it(self):
        graph = nx.MultiDiGraph()
        graph.add_node("e1", node_context_embedding=[0.1, 0.2])
        graph.graph["source"] = "test"

        with tempfile.TemporaryDirectory() as directory:
            store = NetworkXGraphStore(Path(directory) / "graphs")
            path = store.save(graph, "pubmed 0")
            loaded = store.load("pubmed 0")

            self.assertEqual(path.name, "pubmed_0.gpickle")
            self.assertEqual(store.list(), (path,))
            self.assertEqual(loaded.nodes["e1"]["node_context_embedding"], [0.1, 0.2])
            self.assertEqual(loaded.graph["source"], "test")

    def test_normalization_can_be_disabled(self):
        graph = nx.MultiDiGraph()
        graph.add_node("e1")
        context = type("Context", (), {"entity_id": "e1", "summary": "summary"})()

        NodeContextEmbedder(
            FakeEmbeddingModel(),
            NodeEmbeddingConfig(normalize=False),
        ).enrich_graph(graph, (context,))

        self.assertEqual(graph.nodes["e1"]["node_context_embedding"], [3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
