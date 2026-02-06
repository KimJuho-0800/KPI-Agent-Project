from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import os

import pandas as pd

try:
    # supabase-py (PyPI name: supabase)
    from supabase import create_client
    from supabase.client import Client as SupabaseClient
except ImportError as e:
    raise ImportError(
        "supabase-py가 필요합니다. `pip install supabase` 로 설치한 뒤 다시 실행하세요."
    ) from e


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str


class SupabaseDatabase:
    """
    .env에서 SUPABASE_URL, SUPABASE_KEY를 읽어 Supabase에 연결하고,
    검증된 KPI DataFrame을 kpi_data 테이블에 chunk 단위로 insert합니다.
    """

    def __init__(
        self,
        env_path: str | Path = ".env",
        table_name: str = "kpi_data",
        chunk_size: int = 50,
    ) -> None:
        self.env_path = Path(env_path)
        self.table_name = table_name
        self.chunk_size = int(chunk_size)

        self._load_env(self.env_path)
        self.config = self._read_config()
        self.client: SupabaseClient = create_client(self.config.url, self.config.key)

    def insert_kpi_dataframe(self, df: pd.DataFrame) -> int:
        """
        validator.py에서 검증 완료된 DataFrame을 받아 kpi_data 테이블에 bulk insert합니다.
        너무 큰 payload를 피하기 위해 chunk_size(기본 50) 단위로 나누어 넣습니다.

        반환값은 삽입 성공한 총 row 수입니다.
        """
        if df is None:
            raise ValueError("df가 None입니다.")
        if df.empty:
            return 0

        payload = self._df_to_records(df)
        total = len(payload)

        inserted = 0
        for chunk_idx, chunk in enumerate(self._chunk(payload, self.chunk_size)):
            start = chunk_idx * self.chunk_size
            end = min(start + len(chunk), total)  # end는 실 row 기준

            try:
                resp = self.client.table(self.table_name).insert(chunk).execute()
                # supabase-py의 응답 구조는 버전에 따라 다를 수 있으나, 실패 시 예외가 나는 경우가 많습니다.
                # 예외가 나지 않는 케이스를 대비해 error 속성도 점검합니다.
                if hasattr(resp, "error") and resp.error:
                    raise RuntimeError(str(resp.error))
            except Exception as e:
                raise RuntimeError(
                    f"Supabase insert 실패: table={self.table_name}, chunk_idx={chunk_idx}, rows={start}..{end - 1} (size={len(chunk)}). 원인: {e}"
                ) from e

            inserted += len(chunk)

        return inserted

    def _df_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        DataFrame을 Supabase insert용 list[dict]로 변환합니다.
        - date 컬럼이 있으면 ISO 문자열(YYYY-MM-DD 또는 ISO8601)로 정규화합니다.
        - NaN/NaT는 None으로 변환합니다.
        """
        work = df.copy()

        if "date" in work.columns:
            # 문자열/datetime/date 모두 대응: pandas to_datetime 후 date로 떨어뜨리고 ISO로 변환
            dt = pd.to_datetime(work["date"], errors="coerce")
            if dt.isna().any():
                bad = int(dt.isna().sum())
                raise ValueError(f"date 컬럼에 파싱 불가 값이 있습니다. count={bad}")
            # date만 저장하고 싶으면 .dt.date, timestamp까지면 .dt.strftime 사용
            work["date"] = dt.dt.date.astype(str)

        # NaN -> None
        work = work.where(pd.notna(work), None)

        records: List[Dict[str, Any]] = work.to_dict(orient="records")

        # 혹시라도 numpy 타입이 남아 직렬화 문제를 만드는 경우를 대비해 파이썬 기본형으로 정리
        cleaned: List[Dict[str, Any]] = []
        for r in records:
            row: Dict[str, Any] = {}
            for k, v in r.items():
                if isinstance(v, (pd.Timestamp,)):
                    row[k] = v.isoformat()
                else:
                    row[k] = v
            cleaned.append(row)

        return cleaned

    @staticmethod
    def _chunk(items: Sequence[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
        if size <= 0:
            raise ValueError("chunk_size는 1 이상이어야 합니다.")
        return [list(items[i : i + size]) for i in range(0, len(items), size)]

    def _read_config(self) -> SupabaseConfig:
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()

        if not url or not key:
            raise ValueError(
                "SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다. "
                "`.env` 또는 환경변수에 값을 설정하세요."
            )

        return SupabaseConfig(url=url, key=key)

    def _load_env(self, env_path: Path) -> None:
        """
        python-dotenv가 있으면 load_dotenv를 사용하고, 없으면 .env를 직접 파싱합니다.
        """
        if not env_path.exists():
            # .env가 없어도 OS 환경변수로 주입되었을 수 있으니 바로 실패시키진 않되,
            # 이후 _read_config에서 최종 검증합니다.
            return

        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(dotenv_path=env_path, override=False)
            return
        except Exception:
            # fallback: 최소한의 KEY=VALUE 파싱
            content = env_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


if __name__ == "__main__":
    # 간단 동작 테스트 예시(실제 사용 시 validator.py에서 df를 넘겨주면 됩니다.)
    db = SupabaseDatabase(env_path=".env", table_name="kpi_data", chunk_size=50)
    # df = pd.read_csv("manufacturing_kpi.csv")  # 필요 시 로드
    # inserted = db.insert_kpi_dataframe(df)
    # print(f"Inserted rows: {inserted}")
    print("SupabaseDatabase is ready.")
