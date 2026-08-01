import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from graphicalizer import (
    Ontology,
    OpenAISubgraphNarrator,
    SubgraphNarrativeConfig,
    SubgraphNarrativePrompt,
    sample_random_connected_subgraph,
    subgraph_to_payload,
)


class FakeResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed


class FakeResponses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(
            kwargs["text_format"](
                narrative="The sampled entities are connected by the stated relation.",
                uncertainty="",
                notes="",
            )
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


class SubgraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.MultiDiGraph()
        for index in range(1, 6):
            self.graph.add_node(
                f"e{index}",
                ontology_type="thing",
                raw_mention=f"entity {index}",
                context={"summary": f"context {index}"},
            )
        for index in range(1, 5):
            self.graph.add_edge(
                f"e{index}",
                f"e{index + 1}",
                key=f"r{index}",
                ontology_type="links",
                raw_relation="connects",
            )

    def test_seeded_random_walk_is_connected_and_reproducible(self):
        first = sample_random_connected_subgraph(self.graph, num_nodes=3, seed=17)
        second = sample_random_connected_subgraph(self.graph, num_nodes=3, seed=17)

        self.assertEqual(list(first.nodes), list(second.nodes))
        self.assertTrue(nx.is_connected(first.to_undirected()))
        self.assertEqual(first.number_of_nodes(), 3)

    def test_payload_preserves_node_and_edge_attributes(self):
        subgraph = sample_random_connected_subgraph(self.graph, num_nodes=3, seed=17)
        payload = subgraph_to_payload(subgraph)

        self.assertTrue(all("attributes" in node for node in payload["nodes"]))
        self.assertTrue(all("attributes" in edge for edge in payload["edges"]))
        self.assertEqual(len(payload["nodes"]), 3)

    def test_narrative_uses_requested_effort_and_serialized_prompt(self):
        ontology = Ontology(
            entity_types={"thing": {}},
            relation_types={"links": {}},
        )
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "narrative.yaml"
            narrator = OpenAISubgraphNarrator(
                ontology,
                model="test-model",
                prompt=SubgraphNarrativePrompt.default(),
                client=client,
                prompt_snapshot_path=snapshot,
            )
            subgraph = sample_random_connected_subgraph(
                self.graph,
                num_nodes=3,
                seed=17,
            )
            result = narrator.narrate(
                subgraph,
                SubgraphNarrativeConfig(target_words=42),
            )

            self.assertEqual(result.requested_words, 42)
            self.assertEqual(result.actual_words, 9)
            self.assertTrue(snapshot.exists())
            self.assertEqual(client.responses.calls[0]["model"], "test-model")
            self.assertNotIn("42", client.responses.calls[0]["input"][0]["content"])
            self.assertEqual(
                json.loads(client.responses.calls[0]["input"][1]["content"])[
                    "narrative_config"
                ]["target_words"],
                42,
            )


if __name__ == "__main__":
    unittest.main()
