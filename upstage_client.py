"""
upstage_client.py — Upstage Document Parse + Embeddings 클라이언트

Docs: https://console.upstage.ai/docs/getting-started

기능
- DocumentParser : PDF 파일을 Upstage Document Parse API로 파싱하고 디스크에 캐시.
- Embedder      : passage/query 임베딩을 배치 호출 + 디스크 캐시.

표준 라이브러리만 사용 (urllib).
"""

from __future__ import annotations

import io
import json
import math
import os
import ssl
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from pypdf import PdfReader, PdfWriter

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


UPSTAGE_BASE = "https://api.upstage.ai/v1"
DOC_PARSE_URL = f"{UPSTAGE_BASE}/document-digitization"
EMBED_URL = f"{UPSTAGE_BASE}/embeddings"

DOC_PARSE_MODEL = "document-parse"
EMBED_PASSAGE_MODEL = "embedding-passage"
EMBED_QUERY_MODEL = "embedding-query"

CACHE_DIR = Path(".cache")


# ─── multipart/form-data (urllib에 내장 X) ───────────────────────────────────

def _build_multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = "----py_" + os.urandom(8).hex()
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        parts.append(f"{v}\r\n".encode())
    for k, (fname, content, ctype) in files.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def _api_key() -> str:
    key = os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise EnvironmentError(
            "UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다. "
            "source set_env.sh 로 키를 설정하세요."
        )
    return key


# ─── 429 / 5xx 재시도 ───────────────────────────────────────────────────────

_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES  = 6  # 5 → 10 → 20 → 40 → 80 → 120s (cap)


def _build_ssl_context():
    return ssl.create_default_context(cafile=certifi.where() if certifi is not None else None)


def _urlopen_with_retry(req: urllib.request.Request, timeout: int, label: str) -> dict:
    """429/5xx 시 Retry-After 또는 exponential backoff 으로 재시도하고 JSON 반환."""
    backoff = 5
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_build_ssl_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code not in _RETRY_STATUS or attempt == _MAX_RETRIES:
                raise RuntimeError(f"{label} 실패 [{e.code}]: {body}") from e
            wait_s = _retry_after(e) or backoff
            print(
                f"    [{label}] HTTP {e.code} (attempt {attempt}/{_MAX_RETRIES}) "
                f"— {wait_s}s 후 재시도"
            )
            time.sleep(wait_s)
            backoff = min(backoff * 2, 120)
        except urllib.error.URLError as e:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(f"{label} 네트워크 실패: {e}") from e
            print(f"    [{label}] URLError (attempt {attempt}) — {backoff}s 후 재시도: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
    raise RuntimeError(f"{label} 실패 (재시도 소진)")


def _retry_after(e: urllib.error.HTTPError) -> int | None:
    try:
        v = e.headers.get("Retry-After")
        return int(v) if v else None
    except (TypeError, ValueError):
        return None


# ─── Document Parse ─────────────────────────────────────────────────────────

class DocumentParser:
    """Upstage Document Parse 호출 + 결과 캐싱.

    Document Parse 싱크 엔드포인트는 호출당 페이지 수 제한이 있으므로
    PDF 를 PAGES_PER_CHUNK 단위로 쪼개 순차 호출하고 결과 텍스트를 결합한다.
    각 청크는 별도로 캐시돼 중간 실패 후 재실행에서 이어진다.
    """

    PAGES_PER_CHUNK = 80  # 싱크 API 한도(~100p) 아래 안전 마진

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_path: str | Path) -> dict:
        pdf_path = Path(pdf_path)
        file_hash = _file_hash(pdf_path)
        merged_path = self.cache_dir / f"parsed_{file_hash}.json"
        if merged_path.exists():
            print(f"  [parse-cache] {pdf_path.name} → {merged_path.name}")
            return json.loads(merged_path.read_text(encoding="utf-8"))

        reader = PdfReader(str(pdf_path))
        n_pages = len(reader.pages)
        n_chunks = math.ceil(n_pages / self.PAGES_PER_CHUNK)
        print(
            f"  [parse] {pdf_path.name} ({pdf_path.stat().st_size/1e6:.1f} MB, "
            f"{n_pages} pages) → {n_chunks} chunks × {self.PAGES_PER_CHUNK}p"
        )

        text_parts: list[str] = []
        for i in range(n_chunks):
            start_p = i * self.PAGES_PER_CHUNK
            end_p = min((i + 1) * self.PAGES_PER_CHUNK, n_pages)
            chunk_cache = self.cache_dir / f"parsed_{file_hash}_{i:03d}.json"

            if chunk_cache.exists():
                chunk_result = json.loads(chunk_cache.read_text(encoding="utf-8"))
                print(f"    [{i+1:3d}/{n_chunks}] pages {start_p+1}-{end_p}: cached")
            else:
                chunk_bytes = self._extract_pages(reader, start_p, end_p)
                fname = f"{pdf_path.stem}_p{start_p+1:05d}-{end_p:05d}.pdf"
                t0 = time.perf_counter()
                chunk_result = self._call_parse_api(chunk_bytes, fname)
                chunk_cache.write_text(
                    json.dumps(chunk_result, ensure_ascii=False), encoding="utf-8"
                )
                print(
                    f"    [{i+1:3d}/{n_chunks}] pages {start_p+1}-{end_p}: "
                    f"{time.perf_counter()-t0:.1f}s → cached"
                )

            text_parts.append(extract_text_from_parse(chunk_result))

        merged = {"content": {"text": "\n\n".join(t for t in text_parts if t.strip())}}
        merged_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        return merged

    @staticmethod
    def _extract_pages(reader: PdfReader, start: int, end: int) -> bytes:
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    @staticmethod
    def _call_parse_api(pdf_bytes: bytes, filename: str) -> dict:
        body, boundary = _build_multipart(
            fields={
                "model": DOC_PARSE_MODEL,
                "ocr": "auto",
                "output_formats": '["text"]',
                "coordinates": "false",
            },
            files={"document": (filename, pdf_bytes, "application/pdf")},
        )
        req = urllib.request.Request(
            url=DOC_PARSE_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        return _urlopen_with_retry(req, timeout=600, label=f"Document Parse {filename}")


# ─── Embeddings ─────────────────────────────────────────────────────────────

class Embedder:
    """passage/query 임베딩 + 결과 캐싱 (passage만 캐시)."""

    def __init__(self, cache_dir: Path = CACHE_DIR, batch_size: int = 100) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

    def embed_passages(self, texts: list[str], cache_key: str) -> np.ndarray:
        """배치별로 디스크에 저장하면서 임베딩. 중간 실패 후 재실행은 캐시된 배치를 건너뛴다."""
        emb_path = self.cache_dir / f"emb_{cache_key}.npy"
        if emb_path.exists():
            print(f"  [embed-cache] passages → {emb_path}")
            return np.load(emb_path)

        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        print(f"  [embed] {len(texts)} passages × {self.batch_size}/batch = {n_batches} batches")

        batch_dir = self.cache_dir / f"emb_{cache_key}_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)

        vecs: list[np.ndarray] = []
        for bi in range(n_batches):
            batch_path = batch_dir / f"{bi:05d}.npy"
            if batch_path.exists():
                v = np.load(batch_path)
                print(f"    batch {bi+1}/{n_batches}: cached")
            else:
                batch = texts[bi * self.batch_size : (bi + 1) * self.batch_size]
                v = self._embed_batch(batch, EMBED_PASSAGE_MODEL)
                np.save(batch_path, v)
                print(f"    batch {bi+1}/{n_batches}: {len(batch)} texts → saved")
            vecs.append(v)

        out = np.vstack(vecs).astype(np.float32)
        np.save(emb_path, out)
        print(f"  [embed] merged → {emb_path.name} ({out.shape})")
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text], EMBED_QUERY_MODEL)[0]

    def _embed_batch(self, texts: list[str], model: str) -> np.ndarray:
        payload = {"model": model, "input": texts}
        req = urllib.request.Request(
            url=EMBED_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
        )
        data = _urlopen_with_retry(req, timeout=120, label=f"Embedding {model}")
        return np.array([d["embedding"] for d in data["data"]], dtype=np.float32)


# ─── helpers ────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def extract_text_from_parse(parsed: dict) -> str:
    """Document Parse 응답에서 전체 텍스트를 추출.

    응답 스키마는 모델 버전에 따라 다를 수 있어 여러 키를 시도한다."""
    content = parsed.get("content")
    if isinstance(content, dict):
        for key in ("text", "markdown", "html"):
            if content.get(key):
                return content[key]
    if isinstance(content, str):
        return content
    elements = parsed.get("elements") or []
    pieces: list[str] = []
    for el in elements:
        c = el.get("content")
        if isinstance(c, dict):
            pieces.append(c.get("text") or c.get("markdown") or c.get("html") or "")
        elif isinstance(c, str):
            pieces.append(c)
    return "\n\n".join(p for p in pieces if p.strip())
