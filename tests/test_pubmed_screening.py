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
    normalize_screening_output,
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
    def complete(self, prompt, **kwargs):
        raise RuntimeError("insufficient_quota")


class PubMedScreeningTests(unittest.TestCase):
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
