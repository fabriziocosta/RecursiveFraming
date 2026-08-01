import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from graphicalizer import (
    PathogenSpec,
    PubMedArticle,
    build_pathogen_search_query,
    collect_pubmed_corpus,
    derive_category_counts,
    classify_abstract,
    load_search_bundle,
    load_corpus_articles,
    load_refinement_run,
    normalize_screening_output,
    normalize_refinement_output,
    refine_pubmed_corpus,
    build_refinement_prompt,
    build_screening_prompt,
    screen_pubmed_corpus,
    TerminalLLMError,
)


class FakePubMedClient:
    def __init__(self):
        self.search_calls = []
        self.fetch_calls = []
        self.articles = {
            "1": PubMedArticle("1", "Animal study", "Animal infection in bats.", publication_date="2020"),
            "2": PubMedArticle("2", "Zoonosis study", "Nipah virus caused human infection.", publication_date="2022"),
            "3": PubMedArticle("3", "Animal follow-up", "Animal infection remained under study.", publication_date="2023"),
        }

    def search_all(self, query, *, max_results=None, page_size=1000):
        search_type = "zoonosis" if "zoonosis" in query or "human infection" in query else "animal"
        self.search_calls.append((search_type, query))
        return (2, ["1", "2"] if search_type == "animal" else ["2", "3"])

    def fetch_many(self, pmids, *, batch_size=200):
        self.fetch_calls.append(list(pmids))
        return [self.articles[pmid] for pmid in pmids if pmid in self.articles]


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        payload = json.loads(prompt)
        abstract = payload["abstract"]
        if "human infection" in abstract:
            category = 3
            zoonosis = True
        else:
            category = 1
            zoonosis = False
        return json.dumps(
            {
                "category": category,
                "target_pathogen_supported": True,
                "animal_infection_supported": True,
                "zoonosis_supported": zoonosis,
                "evidence_phrases": [abstract],
                "rationale": "The target pathogen is explicitly linked in the abstract.",
                "confidence": 0.95,
                "review_required": False,
            }
        )


class RetryingLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary provider error")
        return json.dumps(
            {
                "category": 1,
                "target_pathogen_supported": True,
                "animal_infection_supported": True,
                "zoonosis_supported": False,
                "evidence_phrases": ["animal infection"],
                "rationale": "Animal-only evidence.",
                "confidence": 0.9,
                "review_required": False,
            }
        )


class TerminalLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        raise RuntimeError("insufficient_quota")


class RefiningLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        payload = json.loads(prompt)
        accepted = "Nipah virus" in payload["abstract"]
        return json.dumps(
            {
                "keep": accepted,
                "target_pathogen_supported": accepted,
                "evidence_relevant": accepted,
                "evidence_phrases": ["Nipah virus"] if accepted else [],
                "rationale": "The target is explicit." if accepted else "Incidental evidence.",
                "confidence": 0.95,
                "review_required": False,
            }
        )


class InterruptingRefiningLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        return json.dumps(
            {
                "keep": True,
                "target_pathogen_supported": True,
                "evidence_relevant": True,
                "evidence_phrases": ["target"],
                "rationale": "The target is explicit.",
                "confidence": 0.95,
                "review_required": False,
            }
        )


class PubMedScreeningTests(unittest.TestCase):
    def test_variable_request_fields_are_at_the_end_of_cacheable_prompts(self):
        for builder in (build_screening_prompt, build_refinement_prompt):
            first = json.loads(builder("Nipah virus", ["Nipah"], "Title A", "Abstract A"))
            second = json.loads(builder("Nipah virus", ["Nipah"], "Title B", "Abstract B"))
            self.assertEqual(list(first), ["_fixed_instructions", "target_pathogen", "target_aliases", "title", "abstract"])
            self.assertEqual(first["_fixed_instructions"], second["_fixed_instructions"])
            self.assertEqual(first["title"], "Title A")
            self.assertEqual(second["title"], "Title B")

    def test_corpus_loader_filters_pathogen_and_year(self):
        frame = pd.DataFrame(
            [
                {
                    "pathogen": "Nipah virus",
                    "pmid": "1",
                    "abstract": "Animal abstract",
                    "publication_date": "2020 Aug",
                    "fetch_status": "ok",
                    "source_search_types": '["animal"]',
                },
                {
                    "pathogen": "Other virus",
                    "pmid": "2",
                    "abstract": "Other abstract",
                    "publication_date": "2022",
                    "fetch_status": "ok",
                    "source_search_types": '["animal"]',
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus_articles.parquet"
            frame.to_parquet(path, index=False)
            filtered = load_corpus_articles(
                path,
                pathogens=["Nipah virus"],
                start_year=2020,
                end_year=2021,
            )

        self.assertEqual(filtered["pmid"].tolist(), ["1"])
        self.assertEqual(int(filtered.loc[0, "publication_year"]), 2020)

    def test_yaml_search_bundle_is_loaded_and_can_define_custom_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.yaml"
            path.write_text(
                "search_type: environment\n"
                "label: Environmental evidence\n"
                "terms:\n"
                "  - reservoir\n"
                "  - habitat\n",
                encoding="utf-8",
            )
            bundle = load_search_bundle(path, expected_search_type="environment")
            query = build_pathogen_search_query(
                PathogenSpec("Nipah virus"),
                "environment",
                search_bundles={"environment": bundle},
            )

        self.assertEqual(bundle.terms, ("reservoir", "habitat"))
        self.assertIn("reservoir[Title/Abstract]", query)
        self.assertIn("habitat[Title/Abstract]", query)

    def test_refinement_removes_unattributed_abstracts_and_resumes(self):
        client = FakePubMedClient()
        llm = RefiningLLM()
        pathogens = {"Nipah virus": ["NiV"]}
        with tempfile.TemporaryDirectory() as directory:
            corpus = collect_pubmed_corpus(client, pathogens, directory, verbose=False).corpus
            corpus.loc[corpus["pmid"].eq("2"), "abstract"] = "Nipah virus caused human infection."
            first = refine_pubmed_corpus(
                corpus,
                pathogens,
                llm,
                directory,
                model="fake-refiner",
                verbose=False,
            )
            self.assertEqual(len(first.refined_corpus), 1)
            self.assertEqual(first.refined_corpus.iloc[0]["pmid"], "2")
            self.assertEqual(llm.calls, 3)

            second = refine_pubmed_corpus(
                corpus,
                pathogens,
                llm,
                directory,
                model="fake-refiner",
                verbose=False,
            )
            self.assertEqual(len(second.refined_corpus), 1)
            self.assertEqual(llm.calls, 3)
            self.assertTrue((Path(directory) / "refined_corpus_articles.parquet").exists())

    def test_refinement_normalization_marks_ambiguous_keep_for_review(self):
        result = normalize_refinement_output(
            {
                "keep": True,
                "target_pathogen_supported": False,
                "evidence_relevant": False,
                "confidence": 0.9,
                "review_required": False,
            }
        )
        self.assertTrue(result["review_required"])
        self.assertFalse(result["accepted"])

    def test_refinement_checkpoint_can_be_reconstructed_after_interrupt(self):
        corpus = pd.DataFrame(
            [
                {
                    "config_hash": "cfg",
                    "pathogen": "Coxiella burnetii",
                    "pmid": "1",
                    "title": "One",
                    "abstract": "target",
                    "fetch_status": "ok",
                    "publication_date": "2020",
                },
                {
                    "config_hash": "cfg",
                    "pathogen": "Coxiella burnetii",
                    "pmid": "2",
                    "title": "Two",
                    "abstract": "target",
                    "fetch_status": "ok",
                    "publication_date": "2021",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                refine_pubmed_corpus(
                    corpus,
                    {"Coxiella burnetii": ["Coxiella"]},
                    InterruptingRefiningLLM(),
                    directory,
                    model="fake-refiner",
                    verbose=False,
                )

            restored = load_refinement_run(corpus, directory)
            self.assertEqual(len(restored.refinement), 1)
            self.assertEqual(len(restored.refined_corpus), 1)
            self.assertTrue(restored.manifest["reconstructed_from_checkpoints"])

    def test_query_contains_both_pathogen_and_evidence_clauses(self):
        query = build_pathogen_search_query(
            PathogenSpec("Nipah virus", ("NiV",)),
            "zoonosis",
            zoonosis_terms=("zoonotic", "human infection"),
        )
        self.assertIn('"Nipah virus"[Title/Abstract]', query)
        self.assertIn('NiV[Title/Abstract]', query)
        self.assertIn('"human infection"[Title/Abstract]', query)

    def test_collection_deduplicates_and_resumes(self):
        client = FakePubMedClient()
        pathogens = {"Nipah virus": ["NiV"]}
        with tempfile.TemporaryDirectory() as directory:
            first = collect_pubmed_corpus(
                client,
                pathogens,
                directory,
                animal_terms=("animal",),
                zoonosis_terms=("zoonosis",),
                verbose=False,
            )
            self.assertEqual(set(first.corpus["pmid"]), {"1", "2", "3"})
            self.assertEqual(len(client.search_calls), 2)
            self.assertEqual(len(client.fetch_calls), 1)

            second = collect_pubmed_corpus(
                client,
                pathogens,
                directory,
                animal_terms=("animal",),
                zoonosis_terms=("zoonosis",),
                verbose=False,
            )
            self.assertEqual(set(second.corpus["pmid"]), {"1", "2", "3"})
            self.assertEqual(len(client.search_calls), 2)
            self.assertEqual(len(client.fetch_calls), 1)
            self.assertTrue((Path(directory) / "corpus_articles.parquet").exists())

    def test_screening_resumes_and_category_derivation(self):
        client = FakePubMedClient()
        llm = FakeLLM()
        pathogens = {"Nipah virus": ["NiV"]}
        with tempfile.TemporaryDirectory() as directory:
            corpus = collect_pubmed_corpus(client, pathogens, directory, verbose=False).corpus
            first = screen_pubmed_corpus(corpus, pathogens, llm, directory, model="fake", verbose=False)
            self.assertEqual(llm.calls, 3)
            second = screen_pubmed_corpus(corpus, pathogens, llm, directory, model="fake", verbose=False)
            self.assertEqual(llm.calls, 3)

            enriched, timeline, counts = derive_category_counts(second.screening, directory)
            self.assertEqual(int(timeline.loc[0, "first_confirmed_zoonosis_year"]), 2022)
            self.assertEqual(set(enriched["final_category"].dropna().astype(int)), {2, 3})
            self.assertEqual(set(counts["category"]), {2, 3})
            pathogen_table = pd.read_parquet(Path(directory) / "pathogen_category_table.parquet")
            self.assertEqual(pathogen_table.loc[0, "pathogen"], "Nipah virus")
            self.assertEqual(int(pathogen_table.loc[0, "category_2_count"]), 2)
            self.assertEqual(int(pathogen_table.loc[0, "category_3_count"]), 1)

    def test_screening_does_not_repeat_persisted_terminal_failures(self):
        corpus = pd.DataFrame(
            [
                {
                    "config_hash": "cfg",
                    "pathogen": "Nipah virus",
                    "pmid": "1",
                    "title": "Study",
                    "abstract": "Animal infection.",
                    "fetch_status": "ok",
                    "publication_date": "2020",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            llm = TerminalLLM()
            first = screen_pubmed_corpus(
                corpus, {"Nipah virus": ["NiV"]}, llm, directory, model="fake", verbose=False
            )
            second = screen_pubmed_corpus(
                corpus, {"Nipah virus": ["NiV"]}, llm, directory, model="fake", verbose=False
            )
            self.assertEqual(llm.calls, 1)
            self.assertEqual(len(first.screening), 1)
            self.assertEqual(len(second.screening), 1)
            self.assertEqual(second.screening.iloc[0]["classification_status"], "terminal_failure")

    def test_invalid_or_ambiguous_output_is_reviewable(self):
        result = normalize_screening_output(
            {
                "category": 3,
                "target_pathogen_supported": False,
                "animal_infection_supported": False,
                "zoonosis_supported": True,
                "evidence_phrases": [],
                "rationale": "Ambiguous attribution.",
                "confidence": 0.4,
            }
        )
        self.assertTrue(result["review_required"])
        self.assertEqual(result["llm_category"], 3)

    def test_classifier_retries_and_records_attempt_count(self):
        llm = RetryingLLM()
        result = classify_abstract(
            llm,
            "Nipah virus",
            ("NiV",),
            "Animal study",
            "Animal infection in bats.",
            retries=2,
            backoff_factor=1,
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["classification_status"], "classified")

    def test_terminal_llm_errors_stop_without_retrying(self):
        with self.assertRaises(TerminalLLMError):
            classify_abstract(
                TerminalLLM(),
                "Nipah virus",
                ("NiV",),
                "Animal study",
                "Animal infection in bats.",
                retries=3,
                backoff_factor=1,
            )


if __name__ == "__main__":
    unittest.main()
