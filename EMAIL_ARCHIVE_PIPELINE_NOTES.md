# Email Archive Parsing/Chunking Notes

이 문서는 `baseline_rag.py`의 이메일 아카이브 PDF 전용 ingestion 로직(파싱→청킹)을 설명합니다.
README(주최측 안내)는 수정하지 않고, 개발자용으로 별도 작성했습니다.

## 1) 현재 산출물(artifact) 기준 정보

현재 `artifacts/chunks.preview.jsonl`은 아래 입력으로 생성되었습니다.

- 입력 폴더: `distribution/enron_test`
- 처리된 PDF: `emails_campbell-l.pdf`
- 최근 QA 기준 청크 수: `127`
- 메시지 커버리지: `Message 1 ~ 91` (unique 91)

즉, 현재 preview artifact는 **emails_campbell-l.pdf 기반 결과**입니다.

## 2) 파이프라인 개요

`build_index(corpus_dir)`는 모든 PDF를 이메일 아카이브 형식으로 가정하고 동일하게 처리합니다.

1. Upstage Document Parse API로 PDF page markdown 획득
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
9. embedding text 생성
   - `Subject + From + Sent(있을 때만) + Body`
10. quoted message dedup (`email_hash`) + `archive_refs` 누적
11. `artifacts/chunks.preview.jsonl` 저장

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

## 7) 최근 QA 스냅샷(요약)

- table residue: 해결(0)
- html table residue: 해결(0)
- address quote residue: 해결(0)
- disclaimer 문구 일부 잔존: 소수(예: intended recipient/strictly prohibited 일부)

운영 판단:
- 해커톤 baseline으로 사용 가능한 상태
- 필요 시 disclaimer 룰만 후속 미세 조정 가능

## 8) 재생성 커맨드

```bash
python3 - <<'PY'
from baseline_rag import build_index
idx = build_index("distribution/enron_test")
print(idx["stats"])
PY
```

생성 결과:
- `artifacts/chunks.preview.jsonl`

