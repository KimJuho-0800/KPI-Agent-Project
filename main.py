import pandas as pd
import numpy as np

def generate_manufacturing_kpi_csv(
    output_path: str = "manufacturing_kpi.csv",
    start_date: str = "2026-01-01",
    end_date: str = "2026-03-31",
    seed: int = 42
) -> pd.DataFrame:
    """
    가상의 제조 KPI(날짜, 생산량, 불량률) 데이터를 생성하고 CSV로 저장합니다.
    - 생산량: 요일 효과(주말↓) + 완만한 상승 추세 + 랜덤 노이즈
    - 불량률: 기본값 + 생산량 영향(약한 양의 상관) + 랜덤 노이즈, 0~0.2 범위로 클리핑
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # 요일 효과: 월(0)~일(6). 주말은 생산량 낮게, 평일은 조금 높게
    dow = dates.dayofweek.to_numpy()
    weekday_factor = np.where(dow >= 5, 0.78, 1.0)  # 토/일 생산량 감소

    # 완만한 상승 추세 (예: 라인 안정화로 점진적 증가)
    trend = np.linspace(0, 120, n)

    # 기본 생산량 + 노이즈
    base_output = 900
    noise = rng.normal(loc=0, scale=60, size=n)

    production_qty = (base_output * weekday_factor + trend + noise).round().astype(int)
    production_qty = np.clip(production_qty, 200, None)  # 최소 생산량 보정

    # 불량률: 기본 1.2% + 생산량 영향(미세) + 노이즈
    # production_qty가 높을수록 불량률이 약간 올라가도록 설계
    base_defect = 0.012
    qty_effect = (production_qty - production_qty.mean()) / production_qty.std()
    defect_rate = base_defect + 0.002 * qty_effect + rng.normal(0, 0.0018, n)

    # 범위 제한 (0%~20%)
    defect_rate = np.clip(defect_rate, 0.0, 0.20)

    df = pd.DataFrame({
        "date": dates.date,  # YYYY-MM-DD 형태
        "production_qty": production_qty,
        "defect_rate": np.round(defect_rate, 4)  # 소수점 4자리
    })

    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    df = generate_manufacturing_kpi_csv(
        output_path="manufacturing_kpi.csv",
        start_date="2026-01-01",
        end_date="2026-03-31",
        seed=42
    )
    print(df.head(10))
    print(f"\nSaved to: manufacturing_kpi.csv (rows={len(df)})")
