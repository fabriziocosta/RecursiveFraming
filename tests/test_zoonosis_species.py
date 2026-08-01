import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from graphicalizer import (
    PubMedArticle,
    build_bacterial_species_summary,
    collect_zoonosis_articles,
    run_species_extraction_stage,
)


class FakeZoonosisPubMedClient:
    def search_all(self, query, *, max_results=None, page_size=1000, sort="relevance"):
        return 2, ["1", "2"]

    def fetch_many(self, pmids, *, batch_size=200):
        return [
            PubMedArticle(
                "1",
                "Zoonotic anthrax in cattle",
                "Bacillus anthracis infected cattle and humans.",
                publication_date="2020",
            ),
            PubMedArticle(
                "2",
                "Zoonotic fever",
                "Coxiella burnetii causes zoonotic infection.",
                publication_date="2021",
            ),
        ]


class BoundedFakeZoonosisPubMedClient(FakeZoonosisPubMedClient):
    def search_all(self, query, *, max_results=None, page_size=1000, sort="relevance"):
        return 2, ["1", "2"][:max_results]


class FakeSpeciesLLM:
    def __init__(self, interrupt_on_call=None):
        self.calls = 0
        self.interrupt_on_call = interrupt_on_call
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt
        species = "Bacillus anthracis" if "anthrax" in prompt.lower() else "Coxiella burnetii"
        return json.dumps(
            {
                "associated_with_zoonosis": True,
                "bacterial_species": [species],
                "evidence_phrases": ["zoonotic"],
                "rationale": "The species is explicitly associated with zoonosis.",
                "confidence": 0.9,
                "review_required": False,
            }
        )


class ZoonosisSpeciesTests(unittest.TestCase):
    def test_pubmed_and_species_checkpoints_are_resumable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            articles = collect_zoonosis_articles(
                FakeZoonosisPubMedClient(),
                output_dir,
                max_results=2,
                fetch_batch_size=2,
                verbose=False,
            )
            self.assertEqual(len(articles), 2)
            self.assertTrue((output_dir / "zoonosis_pubmed_articles.parquet").exists())

            title_path = output_dir / "title.parquet"
            with self.assertRaises(KeyboardInterrupt):
                run_species_extraction_stage(
                    articles,
                    FakeSpeciesLLM(interrupt_on_call=2),
                    title_path,
                    stage="title",
                    save_every=10,
                    verbose=False,
                )
            self.assertEqual(len(pd.read_parquet(title_path)), 1)

            title_llm = FakeSpeciesLLM()
            title = run_species_extraction_stage(
                articles,
                title_llm,
                title_path,
                stage="title",
                save_every=10,
                verbose=False,
            )
            abstract = run_species_extraction_stage(
                articles,
                FakeSpeciesLLM(),
                output_dir / "abstract.parquet",
                stage="abstract",
                save_every=10,
                verbose=False,
            )
            summary = build_bacterial_species_summary(title, abstract, output_dir / "summary.parquet")

            self.assertEqual(
                set(summary["bacterial_species"]),
                {"Bacillus anthracis", "Coxiella burnetii"},
            )
            self.assertTrue((output_dir / "summary.parquet").exists())
            prompt_payload = json.loads(title_llm.prompts[0])
            self.assertEqual(
                list(prompt_payload),
                ["_fixed_instructions", "stage", "title", "abstract"],
            )

    def test_changed_search_limit_does_not_reuse_old_article_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            client = BoundedFakeZoonosisPubMedClient()

            first = collect_zoonosis_articles(
                client,
                output_dir,
                max_results=2,
                fetch_batch_size=2,
                verbose=False,
            )
            second = collect_zoonosis_articles(
                client,
                output_dir,
                max_results=1,
                fetch_batch_size=2,
                verbose=False,
            )

            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 1)
            self.assertEqual(second["pmid"].astype(str).tolist(), ["1"])


if __name__ == "__main__":
    unittest.main()
