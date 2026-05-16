"""
upstage_tracker.py — Upstage Solar LLM 호출 추적 및 submission.csv 자동 생성 모듈

최종 답변 생성은 반드시 tracker.chat() 을 통해 Solar LLM 으로 수행해야 합니다.
used_tokens 가 0 인 제출은 채점에서 제외됩니다.

사용법:
    tracker = UpstageTracker()            # UPSTAGE_API_KEY 환경변수 자동 로드
    answer  = tracker.chat(
        question_id = "Q_001",
        messages    = [{"role": "user", "content": "질문 텍스트"}],
        token       = "무결성_토큰값",
    )
    tracker.save_csv("submission.csv")
"""

import os
import time
import json
import ssl
import urllib.request
import urllib.error
import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
DEFAULT_MODEL    = "solar-mini"


def _build_ssl_context():
    return ssl.create_default_context(cafile=certifi.where() if certifi is not None else None)


class UpstageTracker:
    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        """
        Args:
            api_key: Upstage API 키. None 이면 UPSTAGE_API_KEY 환경변수에서 로드.
            model:   기본 사용 모델 (solar-mini / solar-pro)
        """
        self.api_key = api_key or os.environ.get("UPSTAGE_API_KEY")
        self.model   = model
        self.records: list[dict] = []

        if not self.api_key:
            print(
                "[UpstageTracker] 경고: UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.\n"
                "  export UPSTAGE_API_KEY=<your_key>  또는\n"
                "  UpstageTracker(api_key='...') 로 설정하세요."
            )

    # ── Upstage API 직접 호출 ────────────────────────────────────────────

    def chat(
        self,
        question_id: str,
        messages: list[dict],
        token: str,
        model: str = None,
        system_prompt: str = None,
        **kwargs,
    ) -> str:
        """Upstage Solar API를 호출하고 결과를 자동으로 기록합니다.

        Args:
            question_id:   쿼리 ID (예: "Q_001")
            messages:      [{"role": "user", "content": "..."}] 형식의 메시지 목록
            token:         decryptor가 반환한 무결성 검증 토큰
            model:         모델 오버라이드 (기본값: 인스턴스 생성 시 설정한 모델)
            system_prompt: system 메시지를 간편하게 추가할 때 사용
            **kwargs:      temperature, max_tokens 등 API 파라미터 전달

        Returns:
            LLM이 생성한 답변 문자열
        """
        if not self.api_key:
            raise EnvironmentError(
                "UPSTAGE_API_KEY가 설정되지 않았습니다. "
                "UpstageTracker(api_key='...') 또는 환경변수를 설정하세요."
            )

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model":    model or self.model,
            "messages": full_messages,
            **kwargs,
        }

        start   = time.perf_counter()
        raw     = self._call_api(payload)
        elapsed = time.perf_counter() - start

        answer      = raw["choices"][0]["message"]["content"]
        used_tokens = raw["usage"]["total_tokens"]

        self.records.append({
            "question_id":   question_id,
            "answer":        answer,
            "used_tokens":   used_tokens,
            "inference_time": round(elapsed, 3),
            "token":         token,
        })

        return answer

    # ── 결과 저장 ────────────────────────────────────────────────────────

    def save_csv(self, path: str = "submission.csv") -> None:
        """기록된 모든 결과를 submission.csv로 저장합니다."""
        if not self.records:
            print("[UpstageTracker] 저장할 기록이 없습니다.")
            return

        df = pd.DataFrame(self.records)[
            ["question_id", "answer", "used_tokens", "inference_time", "token"]
        ]
        df.to_csv(path, index=False, encoding="utf-8")

        median_time = df["inference_time"].median()
        total_tok   = df["used_tokens"].sum()
        print(
            f"[UpstageTracker] {path} 저장 완료\n"
            f"  기록 수: {len(df)}개 | 중간값 응답: {median_time:.2f}초 | 총 토큰: {total_tok:,}"
        )

    # ── 내부 HTTP 호출 ───────────────────────────────────────────────────

    def _call_api(self, payload: dict) -> dict:
        req = urllib.request.Request(
            url     = f"{UPSTAGE_BASE_URL}/chat/completions",
            data    = json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, context=_build_ssl_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError(f"Upstage API 오류 [{e.code}]: {body}") from e
