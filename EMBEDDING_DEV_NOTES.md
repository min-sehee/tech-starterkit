# Embedding / Retrieval 개발 노트

이 문서는 `baseline_rag.py`의 **임베딩·인덱싱·검색·압축** 구현 현황을 팀 내부 공유용으로 정리한 문서입니다.  
파싱/청킹 담당 문서(`INGESTION_DEV_NOTES.md`)와 연계되며, 해당 문서의 출력 스키마를 입력으로 받습니다.

---

## 1. 구현 범위

`build_index()` 및 `retrieve()` 내에 구현된 단계:

| 단계 | 내용 | 함수 |
|------|------|------|
| Dense 임베딩 | BGE-large 로컬 모델로 청크 벡터화 | `_embed_texts()` |
| 벡터 DB 적재 | ChromaDB (cosine) 에 임베딩 저장 | `build_index()` 내부 |
| BM25 인덱스 | Okapi BM25 키워드 인덱스 생성 | `build_index()` 내부 |
| 압축 메모리 | Solar mini로 각 청크를 2-3문장 요약 (Index-time) | `_batch_compress_chunks()` |
| 캐시 | 위 3종 결과물을 디스크에 저장, 재실행 시 즉시 로드 | `build_index()` 내부 |
| Hybrid Retrieval | BM25 + Dense → RRF 합산 → top-k | `retrieve()` |
| 컨텍스트 조립 | 검색된 청크의 압축본을 LLM 입력으로 조립 | `retrieve()` 내부 |

---

## 2. 핵심 설계 결정

### 2-1. 임베딩 모델: `BAAI/bge-large-en-v1.5`

- **선택 이유**: 실제 테스트 코퍼스가 영어 이메일(Enron)임이 확인됨. MTEB 영어 리더보드 최상위권 모델.
- **쿼리 전용 prefix 필수**: BGE 계열은 retrieval 시 쿼리에만 prefix를 붙여야 성능이 나옴.
  ```
  "Represent this sentence for searching relevant passages: " + <질문>
  ```
  문서 임베딩(인덱스 시)에는 prefix 없이 사용.
- **레이턴시**: 로컬 실행이므로 API 왕복 없음. GPU 있으면 자동 사용.

### 2-2. 벡터 DB: ChromaDB (cosine)

- 15개 PDF × 수백 청크 규모에서 FAISS 대비 실익 없음. ChromaDB가 이미 requirements에 포함.
- `hnsw:space: cosine` — normalize_embeddings=True와 조합하면 내적 = 코사인 유사도.

### 2-3. 하이브리드 검색 + RRF

BM25와 Dense를 독립적으로 top-20씩 뽑은 뒤 **Reciprocal Rank Fusion**으로 합산:

```
RRF_score(doc) = 1/(k + rank_BM25) + 1/(k + rank_Dense)   (k = 60)
```

- BM25: 정확한 키워드 매칭 (From/To 이름, 날짜, 금액 등)
- Dense: 의미론적 유사도 (패러프레이징, 추론 질문)
- RRF: 두 랭킹을 점수 스케일 차이 없이 합산 가능

### 2-4. Index-time 압축 (CoCom A방식)

**문제**: 검색 시 raw 청크를 그대로 LLM에 전달하면 인사말·서명·반복 문구가 포함되어 입력 토큰 낭비 및 노이즈 발생.

**해결**: 인덱스 빌드 시 Solar mini로 각 청크를 미리 압축 요약. 쿼리 레이턴시에 영향 없음.

```
Raw chunk (retrieval용)  →  ChromaDB + BM25  →  검색에 사용
                         ↓
Compressed chunk (생성용) →  .index_compressed.pkl  →  LLM 컨텍스트에 사용
```

압축 프롬프트 요지:
> 이메일을 2-3문장으로 요약. 보존: From/To/날짜/제목/결정사항/수치. 제거: 인사말/서명.

압축 실패 시(API 오류 등) 원본 텍스트를 폴백으로 사용.

---

## 3. 상수 / 설정값

```python
EMBED_MODEL_NAME    = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "
EMBED_BATCH_SIZE    = 64
CHROMA_PERSIST_DIR  = ".index_chroma"
BM25_CACHE_PATH     = ".index_bm25.pkl"
CHUNKS_CACHE_PATH   = ".index_chunks.pkl"
COMPRESSED_CACHE_PATH = ".index_compressed.pkl"
SOLAR_CHAT_URL      = "https://api.upstage.ai/v1/chat/completions"
COMPRESS_MODEL      = "solar-mini"
```

---

## 4. 함수별 설명

### `_get_embed_model() → SentenceTransformer`

BGE 모델을 싱글턴 패턴으로 로딩. 첫 호출 시 HuggingFace에서 다운로드(~1.3GB), 이후 재사용.

### `_embed_texts(texts, is_query=False) → list[list[float]]`

| 파라미터 | 설명 |
|----------|------|
| `texts` | 임베딩할 문자열 리스트 |
| `is_query` | `True`이면 BGE query prefix 자동 prepend |

반환: L2-정규화된 float 리스트 (cosine 연산용)

### `_build_embed_text(chunk) → str`

청크 임베딩 전 텍스트 전처리. 메타데이터 prefix를 붙여 모델이 문서 출처 맥락을 파악하게 함:

```
[source: emails_bailey-s.pdf | section: ROOT]
<청크 본문>
```

### `_compress_chunk_solar(text, api_key) → str | None`

Solar mini 단일 청크 압축 호출. 타임아웃 30초. 실패 시 `None`.

### `_batch_compress_chunks(chunks, api_key) → dict[str, str]`

`ThreadPoolExecutor(max_workers=8)` 병렬 처리로 전체 청크를 압축.  
반환: `{chunk_id: compressed_text}`

### `retrieve(question, index, top_k=5) → str`

1. BM25로 candidate 20개 랭킹
2. BGE 쿼리 임베딩 → ChromaDB로 candidate 20개 랭킹
3. RRF(k=60) 합산 → top-k 선택
4. 각 청크에 대해 `compressed[chunk_id]` 우선, 없으면 raw 텍스트 폴백
5. `[Source: <파일명>]\n<텍스트>` 포맷으로 `---` 구분하여 반환

---

## 5. 캐시 파일 구조

인덱스 빌드 후 생성되는 파일:

```
.index_bm25.pkl         — BM25Okapi 객체 (pickle)
.index_chunks.pkl       — list[dict] 전체 청크 (pickle)
.index_compressed.pkl   — dict[chunk_id, compressed_text] (pickle)
.index_chroma/          — ChromaDB 퍼시스턴트 디렉터리
artifacts/
  chunks.preview.jsonl  — 청킹 결과 디버그용 JSONL
```

캐시 히트 조건: 위 4개 경로가 **모두** 존재할 때. 하나라도 없으면 전체 재빌드.

---

## 6. `build_index()` 반환 스키마

```python
{
    "bm25":       BM25Okapi,           # BM25 인덱스
    "chunks":     list[dict],          # 원본 청크 (text + metadata)
    "compressed": dict[str, str],      # chunk_id -> 압축 요약 텍스트
    "collection": chromadb.Collection, # ChromaDB 컬렉션 객체
}
```

---

## 7. 실행 / 테스트

### 패키지 설치

```bash
pip install sentence-transformers chromadb rank-bm25 numpy
```

### 인덱스 빌드 단독 테스트

```bash
python - <<'PY'
import os; os.environ["UPSTAGE_API_KEY"] = "<your_key>"
from baseline_rag import build_index
idx = build_index("distribution/corpus")
print("청크 수:", len(idx["chunks"]))
print("압축 수:", len(idx["compressed"]))
print("샘플 압축본:", list(idx["compressed"].values())[0])
PY
```

### retrieve 단독 테스트 (캐시 필요)

```bash
python - <<'PY'
import os; os.environ["UPSTAGE_API_KEY"] = "<your_key>"
from baseline_rag import build_index, retrieve
idx = build_index("distribution/corpus")  # 캐시 히트면 즉시
ctx = retrieve("Who sent the email about the alpha schedule?", idx)
print(ctx)
PY
```

### 캐시 초기화 (재빌드 강제)

```bash
# Windows PowerShell
Remove-Item .index_bm25.pkl, .index_chunks.pkl, .index_compressed.pkl -ErrorAction SilentlyContinue
Remove-Item .index_chroma -Recurse -Force -ErrorAction SilentlyContinue
```

---

## 8. 알려진 한계 / TODO

1. **retrieve top_k 고정값**: 현재 기본값 `top_k=5`. Level 3 멀티홉 질문은 더 많은 청크가 필요할 수 있음. `generate_answer()`에서 질문 난이도 판단 후 `top_k=10`으로 호출하는 분기 고려.

2. **Cross-encoder Reranker 미적용**: RRF top-20 → top-5로 바로 줄임. `cross-encoder/ms-marco-MiniLM-L-6-v2` 추가 시 정밀도 향상 가능 (~150ms 레이턴시 추가).

3. **이메일 메타데이터 미활용**: `_build_embed_text()`가 `source/section` prefix만 사용. From/To/Subject/Date를 헤더로 추출해 prefix에 포함하면 검색 품질 향상 여지 있음.

4. **압축 병렬 속도**: `max_workers=8`로 제한. Solar mini API rate limit에 따라 조정 필요. 압축 실패 청크는 원본 텍스트 폴백이므로 안전.

5. **멀티홉 전략 미구현**: `retrieve()`는 단일 패스. Level 3 질문은 "1차 검색 → 중간 결론 추출 → 2차 검색" 패턴이 유효할 수 있음.
