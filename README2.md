# 구현 메모

[baseline_rag.py](/Users/hanjisu/tech-starterkit/baseline_rag.py), [upstage_tracker.py](/Users/hanjisu/tech-starterkit/upstage_tracker.py), [requirements.txt](/Users/hanjisu/tech-starterkit/requirements.txt)를 최종 구현으로 반영했습니다.

구성은 `origin/sehee`와 이전 `jisu` 구현을 섞었습니다.

- `sehee` 쪽에서 가져온 것: 질문 단계의 PII/인젝션 필터, multi-hop 의도
- `jisu` 쪽에서 가져온 것: `PyMuPDF + BM25 + TF-IDF` 로컬 하이브리드 검색, suspicious chunk 후순위화, 보안 프롬프트, 출력 후처리, 실패 시 파이프라인 지속
- 최종 생성은 `tracker.chat()`를 유지하면서 `solar-pro`를 명시 호출하도록 했고, macOS SSL 이슈를 막기 위해 `certifi` 기반 HTTPS context도 넣었습니다

실행 검증도 끝냈습니다.

- `python -m py_compile baseline_rag.py upstage_tracker.py decryptor.py validator.py` 통과
- `UPSTAGE_API_KEY=... python baseline_rag.py` 실제 실행 성공
- `submission.csv` 생성 및 `validator.py` 통과
- 현재 더미 질문셋 기준 결과는 `used_tokens` 총합 `11,407`, 중간 응답시간 `0.48초`
- 출력은 한 줄 평문으로 정리되도록 맞췄고 샘플 응답은 `전략기획부`, `이서연`, `60%`, `2026년 3월 15일`, `정보 없음` 형태로 나옵니다

주의할 점은 하나입니다.

- 지금은 `HACKATHON_KEY`가 없어서 실제 암호화 질문이 아니라 `decryptor.py`의 더미 질문으로 검증한 상태입니다

원하면 다음으로 실제 대회 당일 체크리스트 기준으로 `HACKATHON_KEY`만 들어왔을 때 어떤 순서로 검증하면 되는지까지 정리해드리겠습니다.
