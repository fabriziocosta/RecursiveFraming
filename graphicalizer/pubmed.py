"""Small, dependency-free PubMed E-utilities client for abstract retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_PUBMED_TOOL = "recursive-framing-graphicalizer"


class PubMedError(RuntimeError):
    """Raised when a PubMed E-utilities request or response is invalid."""


@dataclass(frozen=True)
class PubMedArticle:
    """A PubMed record containing a title and, when available, an abstract."""

    pmid: str
    title: str
    abstract: str
    authors: tuple[str, ...] = ()
    journal: str = ""
    publication_date: str = ""
    doi: str = ""
    url: str = ""

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)


class PubMedClient:
    """Search PubMed and fetch abstract records through NCBI E-utilities."""

    def __init__(
        self,
        *,
        email: str,
        tool: str = DEFAULT_PUBMED_TOOL,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_PUBMED_BASE_URL,
        timeout: float = 30.0,
        min_interval: float = 0.34,
    ) -> None:
        if not email.strip() or "@" not in email:
            raise ValueError("email must be a valid contact email for NCBI requests.")
        if not tool.strip() or " " in tool:
            raise ValueError("tool must be a non-empty name without spaces.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative.")
        self.email = email
        self.tool = tool
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_request_at = 0.0

    @staticmethod
    def keyword_query(
        keywords: Sequence[str],
        *,
        operator: str = "AND",
        field: str = "Title/Abstract",
    ) -> str:
        """Build an explicit PubMed query from keyword phrases."""
        terms = [keyword.strip() for keyword in keywords if keyword.strip()]
        if not terms:
            raise ValueError("keywords must contain at least one non-empty term.")
        if operator not in {"AND", "OR"}:
            raise ValueError("operator must be 'AND' or 'OR'.")
        if not field.strip():
            raise ValueError("field must not be empty.")

        def quote(term: str) -> str:
            escaped = term.replace('"', "")
            if " " in escaped:
                return f'"{escaped}"[{field}]'
            return f"{escaped}[{field}]"

        return f" {operator} ".join(quote(term) for term in terms)

    def _request(self, endpoint: str, params: Mapping[str, Any]) -> bytes:
        wait = self.min_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        query = {
            "tool": self.tool,
            "email": self.email,
            **{key: value for key, value in params.items() if value is not None},
        }
        if self.api_key:
            query["api_key"] = self.api_key
        request = Request(
            f"{self.base_url}/{endpoint}?{urlencode(query)}",
            headers={"User-Agent": f"{self.tool}/0.1 ({self.email})"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except Exception as exc:
            raise PubMedError(f"PubMed request failed at {endpoint}: {exc}") from exc
        self._last_request_at = time.monotonic()
        return body

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        sort: str = "relevance",
    ) -> List[str]:
        """Return PMIDs matching a PubMed query."""
        if not query.strip():
            raise ValueError("query must not be empty.")
        if max_results < 1 or max_results > 10000:
            raise ValueError("max_results must be between 1 and 10000.")
        body = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": sort,
            },
        )
        try:
            data = json.loads(body.decode("utf-8"))
            ids = data["esearchresult"]["idlist"]
        except (ValueError, KeyError, TypeError) as exc:
            raise PubMedError("PubMed search returned an invalid JSON response.") from exc
        return [str(pmid) for pmid in ids]

    def fetch(self, pmids: Sequence[str]) -> List[PubMedArticle]:
        """Fetch PubMed records for a sequence of PMIDs."""
        ids = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]
        if not ids:
            return []
        if len(ids) > 200:
            raise ValueError("fetch accepts at most 200 PMIDs per request.")
        body = self._request(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(ids),
                "rettype": "abstract",
                "retmode": "xml",
            },
        )
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise PubMedError("PubMed fetch returned invalid XML.") from exc
        return [self._parse_article(node) for node in root.findall(".//PubmedArticle")]

    def search_and_fetch(
        self,
        keywords: Sequence[str],
        *,
        max_results: int = 10,
        operator: str = "AND",
        field: str = "Title/Abstract",
        sort: str = "relevance",
    ) -> List[PubMedArticle]:
        """Search using keywords and fetch the matching article records."""
        query = self.keyword_query(keywords, operator=operator, field=field)
        return self.fetch(self.search(query, max_results=max_results, sort=sort))

    def save_articles(
        self,
        articles: Sequence[PubMedArticle],
        output_dir: str | Path,
        *,
        manifest_path: Optional[str | Path] = None,
    ) -> List[Path]:
        """Save article text files and an optional JSON provenance manifest."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        manifest: List[Dict[str, Any]] = []
        for article in articles:
            if not article.abstract.strip():
                continue
            slug = self._slug(article.title)
            path = directory / f"pubmed_{article.pmid}_{slug}.txt"
            path.write_text(self._format_article(article), encoding="utf-8")
            paths.append(path)
            manifest.append(
                {
                    **article.to_mapping(),
                    "path": str(path),
                }
            )
        if manifest_path is not None:
            target = Path(manifest_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "tool": self.tool,
                        "query_results": manifest,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return paths

    def download_by_keywords(
        self,
        keywords: Sequence[str],
        output_dir: str | Path,
        *,
        max_results: int = 10,
        operator: str = "AND",
        field: str = "Title/Abstract",
        sort: str = "relevance",
        manifest_path: Optional[str | Path] = None,
    ) -> List[Path]:
        """Search, fetch, and save abstracts selected by keyword phrases."""
        articles = self.search_and_fetch(
            keywords,
            max_results=max_results,
            operator=operator,
            field=field,
            sort=sort,
        )
        return self.save_articles(
            articles,
            output_dir,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _text(element: Optional[ET.Element]) -> str:
        if element is None:
            return ""
        return "".join(element.itertext()).strip()

    @classmethod
    def _parse_article(cls, node: ET.Element) -> PubMedArticle:
        citation = node.find("./MedlineCitation")
        article = citation.find("./Article") if citation is not None else None
        if citation is None or article is None:
            raise PubMedError("PubMed record is missing MedlineCitation or Article.")

        pmid = cls._text(citation.find("./PMID"))
        title = cls._text(article.find("./ArticleTitle"))
        abstract_parts = []
        for section in article.findall("./Abstract/AbstractText"):
            label = section.attrib.get("Label", "").strip()
            text = cls._text(section)
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)

        authors: List[str] = []
        for author in article.findall("./AuthorList/Author"):
            collective = cls._text(author.find("./CollectiveName"))
            if collective:
                authors.append(collective)
                continue
            last = cls._text(author.find("./LastName"))
            fore = cls._text(author.find("./ForeName"))
            name = " ".join(part for part in (fore, last) if part)
            if name:
                authors.append(name)

        journal = cls._text(article.find("./Journal/Title"))
        publication_date = cls._text(article.find("./Journal/JournalIssue/PubDate"))
        doi = ""
        for identifier in node.findall("./PubmedData/ArticleIdList/ArticleId"):
            if identifier.attrib.get("IdType") == "doi":
                doi = cls._text(identifier)
                break

        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract="\n\n".join(abstract_parts),
            authors=tuple(authors),
            journal=journal,
            publication_date=publication_date,
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        )

    @staticmethod
    def _slug(title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        return (slug[:80] or "untitled").rstrip("_")

    @staticmethod
    def _format_article(article: PubMedArticle) -> str:
        metadata = [
            f"Title: {article.title}",
            f"PMID: {article.pmid}",
            f"Journal: {article.journal}",
            f"Publication date: {article.publication_date}",
        ]
        if article.doi:
            metadata.append(f"DOI: {article.doi}")
        if article.authors:
            metadata.append(f"Authors: {', '.join(article.authors)}")
        return "\n".join(metadata) + "\n\nAbstract:\n" + article.abstract.strip() + "\n"


__all__ = ["DEFAULT_PUBMED_BASE_URL", "DEFAULT_PUBMED_TOOL", "PubMedArticle", "PubMedClient", "PubMedError"]
