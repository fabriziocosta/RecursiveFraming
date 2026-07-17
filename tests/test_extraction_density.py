from types import SimpleNamespace

from graphicalizer import (
    ExtractionDensityConfig,
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
