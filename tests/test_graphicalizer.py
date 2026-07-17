import unittest

from graphicalizer import (
    FreeEntity,
    FreeGraphExtraction,
    FreeRelation,
    GraphicalizerConfig,
    LLMGraphicalizer,
    NodeContext,
    Ontology,
    OntologyCasting,
    OntologyEntity,
    OntologyRelation,
    graph_to_dot,
)


class FakeClient:
    def __init__(self, extraction):
        self.extraction = extraction

    def extract_free_graph(self, text):
        return self.extraction

    def cast_to_ontology(self, text, extraction, ontology):
        return OntologyCasting(
            entities=tuple(
                OntologyEntity(entity.entity_id, "thing", entity.mention_text)
                for entity in extraction.entities
            ),
            relations=tuple(
                OntologyRelation(
                    relation.relation_id,
                    relation.source_id,
                    relation.target_id,
                    "links",
                )
                for relation in extraction.relations
            ),
        )

    def enrich_node_contexts(self, text, extraction, casting, ontology, config):
        return tuple(
            NodeContext(entity.entity_id, f"Context for {entity.mention_text}.")
            for entity in extraction.entities
        )


class GraphicalizerTests(unittest.TestCase):
    def setUp(self):
        self.ontology = Ontology(
            entity_types={"thing": {}},
            relation_types={"links": {}},
        )

    def test_configuration_is_serializable(self):
        config = GraphicalizerConfig.from_mapping(
            {
                "model": "test-model",
                "extraction_density": {
                    "entities_per_word": 0.1,
                    "relations_per_entity": 1.0,
                },
                "node_context": {"max_sentences": 2},
            }
        )
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.node_context.max_sentences, 2)

    def test_pipeline_preserves_explicit_stages_and_parallel_relations(self):
        extraction = FreeGraphExtraction(
            entities=(
                FreeEntity("e1", "alpha", "raw"),
                FreeEntity("e2", "beta", "raw"),
            ),
            relations=(
                FreeRelation("r1", "e1", "e2", "first"),
                FreeRelation("r2", "e1", "e2", "second"),
            ),
        )
        result = LLMGraphicalizer(
            FakeClient(extraction),
            self.ontology,
        ).run("alpha and beta")

        self.assertIs(result.raw_extraction, result.normalized_extraction)
        self.assertEqual(result.normalized_graph.number_of_edges(), 2)
        self.assertEqual(len(result.node_contexts), 2)
        self.assertIn("links", graph_to_dot(result.typed_graph))

    def test_largest_component_is_reported(self):
        extraction = FreeGraphExtraction(
            entities=(
                FreeEntity("e1", "alpha", "raw"),
                FreeEntity("e2", "beta", "raw"),
                FreeEntity("e3", "isolated", "raw"),
            ),
            relations=(FreeRelation("r1", "e1", "e2", "links"),),
        )
        result = LLMGraphicalizer(
            FakeClient(extraction),
            self.ontology,
        ).run("alpha beta isolated")

        self.assertEqual(set(result.normalized_graph.nodes), {"e1", "e2"})
        self.assertEqual(result.normalization_report["discarded_entity_ids"], ("e3",))


if __name__ == "__main__":
    unittest.main()
