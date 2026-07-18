import json
import unittest

from graphicalizer import (
    ExtractionDensityConfig,
    FreeEntity,
    FreeGraphExtraction,
    FreeRelation,
    GraphicalizerPrompt,
    GraphicalizerConfig,
    LLMGraphicalizer,
    NodeContext,
    NodeContextConfig,
    OpenAIResponsesClient,
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


class RepairResponses:
    def __init__(self):
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        casting_type = kwargs["text_format"]
        relation_type = "not_in_ontology" if self.calls == 1 else "links"
        return type(
            "Response",
            (),
            {
                "output_parsed": casting_type(
                    entities=[
                        {
                            "id": "e1",
                            "ontology_type": "thing",
                            "canonical_name": "alpha",
                            "attributes": [],
                            "confidence": 1.0,
                        },
                        {
                            "id": "e2",
                            "ontology_type": "thing",
                            "canonical_name": "beta",
                            "attributes": [],
                            "confidence": 1.0,
                        },
                    ],
                    relations=[
                        {
                            "id": "r1",
                            "source_id": "e1",
                            "target_id": "e2",
                            "ontology_type": relation_type,
                            "attributes": [],
                            "confidence": 1.0,
                        }
                    ],
                    notes="",
                )
            },
        )()


class RepairClient:
    def __init__(self):
        self.responses = RepairResponses()


class PartialContextResponses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        payload = json.loads(kwargs["input"][1]["content"])
        requested_ids = [node["id"] for node in payload["nodes"]]
        self.calls.append(requested_ids)
        returned_ids = requested_ids[:1] if len(self.calls) == 1 else requested_ids
        context_type = kwargs["text_format"]
        return type(
            "Response",
            (),
            {
                "output_parsed": context_type(
                    contexts=[
                        {
                            "id": entity_id,
                            "summary": f"Context for {entity_id}.",
                            "evidence": [],
                            "uncertainty": "",
                        }
                        for entity_id in returned_ids
                    ],
                    notes="",
                )
            },
        )()


class PartialContextClient:
    def __init__(self):
        self.responses = PartialContextResponses()


class FakeEmbeddingModel:
    model_name = "test-embeddings"

    def encode(self, texts):
        return [[3.0, 4.0] for _ in texts]


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

    def test_embedding_model_is_explicitly_injected_at_graphicalizer_init(self):
        extraction = FreeGraphExtraction(
            entities=(
                FreeEntity("e1", "alpha", "raw"),
                FreeEntity("e2", "beta", "raw"),
            ),
            relations=(FreeRelation("r1", "e1", "e2", "links"),),
        )
        result = LLMGraphicalizer(
            FakeClient(extraction),
            self.ontology,
            embedding_model=FakeEmbeddingModel(),
        ).run("alpha and beta")

        self.assertEqual(result.graph.nodes["e1"]["node_context_embedding"], [0.6, 0.8])
        self.assertEqual(result.graph.graph["node_context_embeddings"]["model"], "test-embeddings")

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

    def test_openai_casting_retries_invalid_ontology_types(self):
        extraction = FreeGraphExtraction(
            entities=(
                FreeEntity("e1", "alpha", "raw"),
                FreeEntity("e2", "beta", "raw"),
            ),
            relations=(FreeRelation("r1", "e1", "e2", "connects"),),
        )
        client = RepairClient()
        prompt = GraphicalizerPrompt.default().render(
            self.ontology,
            ExtractionDensityConfig(),
            model="test-model",
        )
        llm_client = OpenAIResponsesClient(
            prompt=prompt,
            ontology=self.ontology,
            client=client,
            model="test-model",
            casting_retries=1,
        )

        casting = llm_client.cast_to_ontology("alpha connects beta", extraction, self.ontology)

        self.assertEqual(casting.relations[0].ontology_type, "links")
        self.assertEqual(client.responses.calls, 2)

    def test_openai_context_retries_only_omitted_nodes(self):
        extraction = FreeGraphExtraction(
            entities=(
                FreeEntity("e1", "alpha", "raw"),
                FreeEntity("e2", "beta", "raw"),
                FreeEntity("e3", "gamma", "raw"),
            ),
            relations=(
                FreeRelation("r1", "e1", "e2", "connects"),
                FreeRelation("r2", "e2", "e3", "connects"),
            ),
        )
        casting = OntologyCasting(
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
        client = PartialContextClient()
        prompt = GraphicalizerPrompt.default().render(
            self.ontology,
            ExtractionDensityConfig(),
            model="test-model",
        )
        llm_client = OpenAIResponsesClient(
            prompt=prompt,
            ontology=self.ontology,
            client=client,
            model="test-model",
            node_context_retries=1,
        )

        contexts = llm_client.enrich_node_contexts(
            "alpha connects beta and gamma",
            extraction,
            casting,
            self.ontology,
            NodeContextConfig(),
        )

        self.assertEqual([context.entity_id for context in contexts], ["e1", "e2", "e3"])
        self.assertEqual(client.responses.calls, [["e1", "e2", "e3"], ["e2", "e3"]])


if __name__ == "__main__":
    unittest.main()
