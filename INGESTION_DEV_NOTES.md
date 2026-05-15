# Ingestion 개발 노트 (Parse -> Chunk)

이 문서는 해커톤용 RAG 파이프라인에서 **ingestion 앞단(파싱/청킹)** 구현 현황을 팀 내부 공유용으로 정리한 문서입니다.  
주최 측 안내 문서(`README.md`)는 수정하지 않고, 개발 협업용 맥락만 별도로 기록합니다.

## 1. 구현 범위

현재 `baseline_rag.py`의 `build_index()`는 아래 단계까지 구현되어 있습니다.

1. PDF -> Upstage Document Parse API 호출
2. Markdown 정규화
3. Sanitization pass-through (현재는 미적용)
4. Heading/문단 기반 청킹
5. 짧은 청크 후처리
6. `artifacts/chunks.preview.jsonl` 저장
7. `{"chunks": [...], "stats": {...}}` 반환

주의:
- `retrieve()`는 미구현 상태이며, 다른 담당자가 구현합니다.
- 임베딩/벡터DB 적재는 다른 담당 범위입니다.

## 2. 실행 전 준비

필수 환경변수:
- `UPSTAGE_API_KEY`

설정 예시:
```bash
export UPSTAGE_API_KEY="<your_key>"
```

## 3. 빌드/스모크 테스트

`build_index()` 단독 테스트:
```bash
python3 - <<'PY'
from baseline_rag import build_index
idx = build_index("distribution/corpus")
print(idx["stats"])
print(idx["chunks"][0]["metadata"] if idx["chunks"] else None)
PY
```

산출물 확인:
```bash
ls -lh artifacts/chunks.preview.jsonl
head -n 3 artifacts/chunks.preview.jsonl
```

## 4. 청킹 규칙 요약

기본 파라미터:
- `target_tokens = 700`
- `overlap_tokens = 100`
- `min_chunk_words = 140` (초기 병합 단계)

후처리 규칙:
1. `contains_table=True` 청크는 항상 단독 유지
2. 비표 청크 중 `3~8 words`:
   - 숫자/날짜/금액 핵심 패턴 없으면 드롭
3. 비표 청크 중 `9~19 words`:
   - 같은 `source/page`의 다음 비표 청크에 병합
   - 다음 청크가 표이거나 없으면 이전 비표 청크에 병합

핵심 의도:
- 과도하게 짧은 노이즈 청크를 줄여 retrieval top-k 낭비를 방지
- 표는 구조적 정보가 중요하므로 분해/병합 최소화

## 5. 출력 스키마 (임베딩 담당 전달 계약)

`build_index()` 반환:
```json
{
  "chunks": [
    {
      "text": "...",
      "metadata": {
        "chunk_id": "budget_2026.pdf::c0",
        "chunk_index": 0,
        "source": "budget_2026.pdf",
        "page": 1,
        "section": "1. 예산 총괄 요약",
        "company": "(주)넥스트코어",
        "doc_title": "budget 2026",
        "doc_id": "FIN-2026-BDG-001",
        "doc_type": "budget",
        "contains_table": true,
        "pipeline_stage": "post_parse_post_chunk"
      }
    }
  ],
  "stats": {
    "num_docs": 7,
    "num_chunks": 35,
    "target_tokens": 700,
    "overlap_tokens": 100,
    "min_chunk_words": 140
  }
}
```

`chunks.preview.jsonl`은 위 `chunks[]` 레코드가 줄 단위로 저장된 디버그 산출물입니다.

## 6. 현재 QA 상태 (최근 실행 기준)

- `num_docs=7`
- `num_chunks=35`
- `short(<20 words)=1`
- `3~8 words=0`
- `ROOT section=0`
- metadata 필수키 누락 `0`

해석:
- 임베딩 담당자가 바로 붙일 수 있는 수준의 baseline 품질 확보
- 남은 짧은 청크 1개는 선택적으로 추가 정제 가능

## 7. 알려진 한계 / TODO

1. Sanitization은 현재 pass-through
- Prompt Injection / Hidden Text / PII masking은 후속 작업 필요

2. 토큰 계산은 word-count 근사
- 추후 `tiktoken` 도입 시 더 정밀한 chunk size 제어 가능

3. Upstage Parse 응답 스키마 변화 가능성
- 현재는 유연 파싱으로 대응하지만, API 응답 변경 시 분기 업데이트 필요

## 8. 협업 가이드

임베딩 담당자가 사용할 진입점:
- `index = build_index("distribution/corpus")`
- `chunks = index["chunks"]`

권장:
1. `metadata.chunk_id`를 vector DB `id`로 사용
2. `metadata.source/page/section`을 retrieval provenance에 그대로 사용
3. `contains_table` 기반으로 re-ranking/answer generation 전략 분기 고려

