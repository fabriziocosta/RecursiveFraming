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


@dataclass(frozen=True)
class PubMedSearchPage:
    """One paginated PubMed search response."""

    query: str
    count: int
    retstart: int
    ids: tuple[str, ...]


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
        retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        if not email.strip() or "@" not in email:
            raise ValueError("email must be a valid contact email for NCBI requests.")
        if not tool.strip() or " " in tool:
            raise ValueError("tool must be a non-empty name without spaces.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative.")
        if retries < 1:
            raise ValueError("retries must be at least 1.")
        if backoff_factor < 1:
            raise ValueError("backoff_factor must be at least 1.")
        self.email = email
        self.tool = tool
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self.retries = retries
        self.backoff_factor = backoff_factor
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
        last_error: Exception | None = None
        for attempt in range(self.retries):
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
                self._last_request_at = time.monotonic()
                return body
            except Exception as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_factor**attempt)
        raise PubMedError(f"PubMed request failed at {endpoint}: {last_error}") from last_error

    def search_page(
        self,
        query: str,
        *,
        retstart: int = 0,
        retmax: int = 1000,
        sort: str = "relevance",
    ) -> PubMedSearchPage:
        """Return one page of PMIDs plus PubMed's total result count."""
        if not query.strip():
            raise ValueError("query must not be empty.")
        if retstart < 0:
            raise ValueError("retstart must be non-negative.")
        if retmax < 1 or retmax > 10000:
            raise ValueError("retmax must be between 1 and 10000.")
        body = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retstart": retstart,
                "retmax": retmax,
                "retmode": "json",
                "sort": sort,
            },
        )
        try:
            data = json.loads(body.decode("utf-8"))
            result = data["esearchresult"]
            count = int(result["count"])
            ids = tuple(str(pmid) for pmid in result.get("idlist", []))
        except (ValueError, KeyError, TypeError) as exc:
            raise PubMedError("PubMed search returned an invalid JSON response.") from exc
        return PubMedSearchPage(query, count, retstart, ids)

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
        return list(self.search_page(query, retmax=max_results, sort=sort).ids)

    def search_all(
        self,
        query: str,
        *,
        max_results: int | None = None,
        page_size: int = 1000,
        sort: str = "relevance",
    ) -> tuple[int, List[str]]:
        """Retrieve all available PMIDs, or a bounded prefix, page by page."""
        if max_results is not None and max_results < 1:
            raise ValueError("max_results must be positive when provided.")
        if page_size < 1 or page_size > 10000:
            raise ValueError("page_size must be between 1 and 10000.")
        first = self.search_page(query, retmax=min(page_size, max_results or page_size), sort=sort)
        total = first.count
        target = min(total, max_results) if max_results is not None else total
        ids = list(first.ids[:target])
        for start in range(len(ids), target, page_size):
            page = self.search_page(
                query,
                retstart=start,
                retmax=min(page_size, target - start),
                sort=sort,
            )
            ids.extend(page.ids)
            if not page.ids:
                break
        return total, ids[:target]

    def fetch(self, pmids: Sequence[str]) -> List[PubMedArticle]:
        """Fetch PubMed records for a sequence of PMIDs."""
        ids = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]
        if not ids:
            return []
        if len(ids) > 200:
            return self.fetch_many(ids)
        return self._fetch_batch(ids)

    def _fetch_batch(self, ids: Sequence[str]) -> List[PubMedArticle]:
        if len(ids) > 200:
            raise ValueError("a PubMed fetch batch accepts at most 200 PMIDs.")
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

    def fetch_many(self, pmids: Sequence[str], *, batch_size: int = 200) -> List[PubMedArticle]:
        """Fetch an arbitrary PMID collection in PubMed-sized batches."""
        if batch_size < 1 or batch_size > 200:
            raise ValueError("batch_size must be between 1 and 200.")
        ids = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]
        articles: List[PubMedArticle] = []
        for start in range(0, len(ids), batch_size):
            articles.extend(self._fetch_batch(ids[start : start + batch_size]))
        return articles

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


__all__ = [
    "DEFAULT_PUBMED_BASE_URL",
    "DEFAULT_PUBMED_TOOL",
    "PubMedArticle",
    "PubMedClient",
    "PubMedError",
    "PubMedSearchPage",
]
