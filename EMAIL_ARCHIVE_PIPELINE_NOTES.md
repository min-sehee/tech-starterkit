# Email Archive Parsing/Chunking Notes

이 문서는 `baseline_rag.py`의 이메일 아카이브 PDF 전용 ingestion 로직(파싱→청킹)을 설명합니다.
README(주최측 안내)는 수정하지 않고, 개발자용으로 별도 작성했습니다.

## 1) 현재 산출물(artifact) 기준 정보

현재 `artifacts/chunks.preview.jsonl`은 아래 입력으로 생성되었습니다.

- 입력 폴더: `distribution/enron_test`
- 처리된 PDF: `emails_love-p.pdf`
- 최근 QA 기준 청크 수: `79`
- 메시지 커버리지: `Message 1 ~ 43` (unique 43)

즉, 현재 preview artifact는 **emails_love-p.pdf 기반 결과**입니다.

## 2) 파이프라인 개요

`build_index(corpus_dir)`는 모든 PDF를 이메일 아카이브 형식으로 가정하고 동일하게 처리합니다.

1. `pypdf` 로컬 파서로 PDF page text 추출
   - `PdfReader(...).pages[i].extract_text()`
   - 외부 Parse API 호출 없음
2. page marker를 붙여 full text 생성
   - 형식: `[[PAGE:n]]\n{markdown}`
3. `Message N of M` 경계로 archive entry 분리
4. 각 entry에서 container header 추출
   - `container_subject`, `container_from`, `container_to`, `archive_file_path`
5. entry body를 current/quoted block으로 분리
   - 첫 block: `current_message`
   - 이후 block: `quoted_message`
6. block별 header 파싱(From/Sent/To/Cc/Subject) + fallback sender 보완
7. block별 attachment 추출 (`email_attachments`)
8. clean body 적용
   - table residue 제거
   - attachment marker 제거
   - disclaimer 제거
9. instruction poisoning sanitize 적용
   - 위치: message split 이후, chunk split 이전
   - 룰 파일: `poison_rules.json`
   - 탐지 축: target/obligation/action/scope
   - 처리: high score block 제거, medium score flag-only
10. embedding text 생성
   - `Subject + From + Sent(있을 때만) + Body`
11. quoted message dedup (`email_hash`) + `archive_refs` 누적
12. `artifacts/chunks.preview.jsonl` 저장

배치 처리:
- `PARSER_BATCH_SIZE` 환경변수로 파일 처리 배치 크기 제어 (기본값 `25`)
- 대량 코퍼스에서 진행 로그를 배치 단위로 출력

## 3) 메타데이터 3층 구조

### A. archive/container level
- `archive_message_no`
- `archive_total_messages`
- `archive_file_path`
- `container_subject`
- `container_from`
- `container_to`

### B. thread level
- `thread_id`
- `thread_subject`
- `thread_attachments` (해당 Message block 전체 첨부)

### C. individual email level
- `email_role` (`current_message` | `quoted_message`)
- `email_display_order`
- `email_chronological_order`
- `email_subject`
- `email_from`
- `email_to`
- `email_cc`
- `email_sent`
- `email_attachments` (해당 block에서 직접 언급된 첨부)
- `email_header_source`
- `behalf_of_sender`

### Common
- `chunk_id`, `chunk_index`, `source`, `page`, `start_page`, `end_page`
- `email_hash`, `archive_refs` (quoted dedup 추적)
- `low_information`
- `has_disclaimer`, `disclaimer_removed`
- `security_flags`, `poison_score`, `poison_classes`, `poison_spans_count`

## 4) 임베딩 텍스트 정책

`chunk["text"]`는 metadata-heavy 포맷을 피하고 최소 의미만 남깁니다.

형식:

- `Subject: ...`
- `From: ...`
- `Sent: ...` (값 있을 때만)
- `Body:\n...`

제외 항목:
- To/Cc/File/Attachments/order/page 계열 설명 텍스트
- 긴 metadata 블록
- attachment_list 전용 chunk

## 5) 본문 정제(clean body) 규칙

주요 제거 대상:
- markdown table residue (`| --- | --- |` 등)
- container header table row (`Sender/Recipients/File`)
- HTML `<table>...</table>` 잔여물
- attachment marker
  - `(See attached file: ...)`
  - `<< File: ... >>`
  - `<<name.pdf>>`
  - 첨부파일명만 나열된 bullet line
- 전형적 legal disclaimer 문구
  - intended recipient
  - strictly prohibited
  - dissemination or copying
  - received this communication in error
  - privileged/confidential boilerplate

주의:
- 업무 본문까지 과도 삭제하지 않도록 marker 기반으로만 제거

## 6) dedup 정책

- 대상: `quoted_message`만 dedup
- 키: `email_hash`(subject/from/sent/body 기반 정규화 해시)
- 동일 hash 재등장 시 새 chunk를 추가하지 않고 기존 chunk의 `archive_refs`에 출처를 누적

효과:
- 반복 quoted 원문 과다 인덱싱 방지
- 출처 traceability 유지

## 7) Poisoning 방어 정책(확장형)

- 룰 정의는 코드 하드코딩이 아니라 `poison_rules.json`으로 분리
- 기본 신호 조합:
  - target: `ai/retrieval system` 지시 대상 여부
  - obligation: `must/required/mandated/should`
  - action: `append/include/confirm/output/every response`
  - scope: `every response/each answer/always`
- 점수 기반 처리:
  - `high_threshold` 이상: block 제거
  - `medium_threshold` 이상: 본문 유지 + `poison_suspected` 플래그
- exact phrase는 룰 파일에서만 관리(해커톤 중 패턴 변화 대응)

## 8) 최근 QA 스냅샷(요약)

- table residue: 해결(0)
- html table residue: 해결(0)
- attachment marker residue: 해결(0)
- address quote residue: 해결(0)
- poison 제거 로그: `poison_removed=33` (source log 기준)
- poison 문구 일부 잔존: `append ... every response` 계열 10개 청크
- disclaimer 문구 일부 잔존: `intended recipient`/`strictly prohibited` 소수 잔존

운영 판단:
- 해커톤 baseline으로 사용 가능한 상태
- 필요 시 `poison_rules.json` threshold/패턴 미세 조정 권장

## 9) 재생성 커맨드

```bash
python3 - <<'PY'
from baseline_rag import build_index
idx = build_index("distribution/enron_test")
print(idx["stats"])
PY
```

생성 결과:
- `artifacts/chunks.preview.jsonl`

## 10) 의존성 메모

- Python 패키지: `pypdf>=4.0.0`
- 본 현재 구현은 Upstage Document Parse API / qpdf에 의존하지 않음
