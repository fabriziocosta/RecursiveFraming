from types import SimpleNamespace

from graphicalizer import (
    ExtractionDensityConfig,
    FreeEntity,
    FreeGraphExtraction,
    FreeRelation,
    GraphicalizerPrompt,
    OpenAIResponsesClient,
    Ontology,
)


class DensityResponses:
    def __init__(self):
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        response_type = kwargs["text_format"]
        entities = [
            {
                "id": "e1",
                "mention_text": "alpha",
                "free_type": "thing",
                "description": "",
                "evidence": "alpha",
                "confidence": 1.0,
            }
        ]
        relations = []
        if self.calls > 1:
            entities.append(
                {
                    "id": "e2",
                    "mention_text": "beta",
                    "free_type": "thing",
                    "description": "",
                    "evidence": "beta",
                    "confidence": 1.0,
                }
            )
            relations.append(
                {
                    "id": "r1",
                    "source_id": "e1",
                    "target_id": "e2",
                    "free_type": "links",
                    "evidence": "alpha beta",
                    "confidence": 1.0,
                }
            )
        return SimpleNamespace(
            output_parsed=response_type(
                entities=entities,
                relations=relations,
                notes="",
            )
        )


class InvalidExpansionResponses(DensityResponses):
    def parse(self, **kwargs):
        if self.calls == 0:
            return super().parse(**kwargs)
        if self.calls == 1:
            self.calls += 1
            response_type = kwargs["text_format"]
            return SimpleNamespace(
                output_parsed=response_type(
                    entities=[
                        {
                            "id": "e1",
                            "mention_text": "alpha",
                            "free_type": "thing",
                            "description": "",
                            "evidence": "alpha",
                            "confidence": 1.0,
                        },
                        {
                            "id": "e2",
                            "mention_text": "beta",
                            "free_type": "thing",
                            "description": "",
                            "evidence": "beta",
                            "confidence": 1.0,
                        },
                    ],
                    relations=[
                        {
                            "id": "r1",
                            "source_id": "e1",
                            "target_id": "e34",
                            "free_type": "links",
                            "evidence": "alpha beta",
                            "confidence": 1.0,
                        }
                    ],
                    notes="",
                )
            )
        return super().parse(**kwargs)


class UnknownCastingResponses:
    def parse(self, **kwargs):
        response_type = kwargs["text_format"]
        return SimpleNamespace(
            output_parsed=response_type(
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
                        "ontology_type": "influences",
                        "attributes": [],
                        "confidence": 1.0,
                    }
                ],
                notes="",
            )
        )


def test_density_retry_expands_an_undersized_extraction():
    ontology = Ontology(entity_types={"thing": {}}, relation_types={"links": {}})
    density = ExtractionDensityConfig(
        entities_per_word=0.2,
        relations_per_entity=0.5,
        minimum_entity_fraction=1.0,
        density_retries=1,
    )
    responses = DensityResponses()
    prompt = GraphicalizerPrompt.default().render(
        ontology,
        density,
        model="test-model",
    )
    client = OpenAIResponsesClient(
        prompt=prompt,
        ontology=ontology,
        extraction_density=density,
        model="test-model",
        client=SimpleNamespace(responses=responses),
    )

    extraction = client.extract_free_graph("one two three four five six seven eight nine ten")

    assert responses.calls == 2
    assert len(extraction.entities) == 2
    assert len(extraction.relations) == 1


def test_density_target_exposes_minimum_count():
    density = ExtractionDensityConfig(
        entities_per_word=0.2,
        minimum_entity_fraction=0.75,
    )

    assert density.target_counts("one two three four five ten")["target_entity_count"] == 1
    assert density.target_counts("one two three four five ten")["minimum_entity_count"] == 1


def test_density_retry_rejects_relations_with_unknown_endpoints():
    ontology = Ontology(entity_types={"thing": {}}, relation_types={"links": {}})
    density = ExtractionDensityConfig(
        entities_per_word=0.2,
        relations_per_entity=0.5,
        minimum_entity_fraction=1.0,
        density_retries=2,
    )
    responses = InvalidExpansionResponses()
    prompt = GraphicalizerPrompt.default().render(
        ontology,
        density,
        model="test-model",
    )
    client = OpenAIResponsesClient(
        prompt=prompt,
        ontology=ontology,
        extraction_density=density,
        model="test-model",
        client=SimpleNamespace(responses=responses),
    )

    extraction = client.extract_free_graph("one two three four five six seven eight nine ten")

    assert responses.calls == 3
    assert len(extraction.entities) == 2
    assert extraction.relations[0].target_id == "e2"


def test_casting_falls_back_for_persistent_unknown_relation_type():
    ontology = Ontology(
        entity_types={"thing": {}, "investigation_entity": {}},
        relation_types={"related_to": {}},
    )
    density = ExtractionDensityConfig(density_retries=0)
    prompt = GraphicalizerPrompt.default().render(
        ontology,
        density,
        model="test-model",
    )
    client = OpenAIResponsesClient(
        prompt=prompt,
        ontology=ontology,
        extraction_density=density,
        model="test-model",
        client=SimpleNamespace(responses=UnknownCastingResponses()),
        casting_retries=0,
    )
    extraction = FreeGraphExtraction(
        entities=(FreeEntity("e1", "alpha", "thing"), FreeEntity("e2", "beta", "thing")),
        relations=(FreeRelation("r1", "e1", "e2", "influences"),),
    )

    casting = client.cast_to_ontology("alpha influences beta", extraction, ontology)

    assert casting.relations[0].ontology_type == "related_to"
    assert casting.relations[0].attributes["unmapped_ontology_type"] == "influences"
