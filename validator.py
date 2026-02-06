from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import pandas as pd


@dataclass
class ValidationResult:
    total_rows: int
    prod_qty_invalid_count: int
    defect_rate_outlier_count: int
    prod_qty_invalid_rows: pd.DataFrame
    defect_rate_outlier_rows: pd.DataFrame

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append("Manufacturing KPI Validation Report")
        lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
        lines.append("-" * 60)
        lines.append(f"Total rows: {self.total_rows}")
        lines.append(f"Invalid production_qty (<= 0): {self.prod_qty_invalid_count}")
        lines.append(f"Outlier defect_rate (> 0.15): {self.defect_rate_outlier_count}")
        lines.append("")

        if self.prod_qty_invalid_count == 0 and self.defect_rate_outlier_count == 0:
            lines.append("Result: PASS (no issues found)")
            return "\n".join(lines)

        lines.append("Result: FAIL (issues found)")
        lines.append("")

        if self.prod_qty_invalid_count > 0:
            lines.append("[Invalid production_qty (<= 0)]")
            for _, r in self.prod_qty_invalid_rows.iterrows():
                lines.append(
                    f"- date={r['date']} | production_qty={r['production_qty']} | defect_rate={r['defect_rate']}"
                )
            lines.append("")

        if self.defect_rate_outlier_count > 0:
            lines.append("[Outlier defect_rate (> 0.15)]")
            for _, r in self.defect_rate_outlier_rows.iterrows():
                lines.append(
                    f"- date={r['date']} | production_qty={r['production_qty']} | defect_rate={r['defect_rate']}"
                )
            lines.append("")

        return "\n".join(lines)


class ManufacturingKPIValidator:
    def __init__(
        self,
        csv_path: str | Path = "manufacturing_kpi.csv",
        report_path: str | Path = "validation_report.txt",
        defect_rate_threshold: float = 0.15,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.report_path = Path(report_path)
        self.defect_rate_threshold = float(defect_rate_threshold)

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path.resolve()}")

        df = pd.read_csv(self.csv_path)

        required_cols = {"date", "production_qty", "defect_rate"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # date 파싱(실패하면 NaT/예외가 아니라 문자열 그대로일 수 있으니 강제)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        if df["date"].isna().any():
            bad_count = int(df["date"].isna().sum())
            raise ValueError(f"Invalid date values found (count={bad_count}). Please fix 'date' column.")

        # 수치형 강제 변환
        df["production_qty"] = pd.to_numeric(df["production_qty"], errors="coerce")
        df["defect_rate"] = pd.to_numeric(df["defect_rate"], errors="coerce")

        if df["production_qty"].isna().any() or df["defect_rate"].isna().any():
            bad_prod = int(df["production_qty"].isna().sum())
            bad_def = int(df["defect_rate"].isna().sum())
            raise ValueError(
                f"Non-numeric values found: production_qty NaN={bad_prod}, defect_rate NaN={bad_def}"
            )

        return df

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        prod_qty_invalid = df[df["production_qty"] <= 0].copy()
        defect_rate_outlier = df[df["defect_rate"] > self.defect_rate_threshold].copy()

        # 보기 좋게 정렬
        prod_qty_invalid = prod_qty_invalid.sort_values("date")
        defect_rate_outlier = defect_rate_outlier.sort_values("date")

        return ValidationResult(
            total_rows=len(df),
            prod_qty_invalid_count=len(prod_qty_invalid),
            defect_rate_outlier_count=len(defect_rate_outlier),
            prod_qty_invalid_rows=prod_qty_invalid,
            defect_rate_outlier_rows=defect_rate_outlier,
        )

    def save_report(self, result: ValidationResult) -> None:
        text = result.to_text()
        self.report_path.write_text(text, encoding="utf-8")

    def run(self) -> ValidationResult:
        df = self.load()
        result = self.validate(df)

        # 콘솔 출력(이상치가 있으면 상세 출력)
        report_text = result.to_text()
        print(report_text)

        # 파일 저장
        self.save_report(result)
        return result


if __name__ == "__main__":
    validator = ManufacturingKPIValidator(
        csv_path="manufacturing_kpi.csv",
        report_path="validation_report.txt",
        defect_rate_threshold=0.15,
    )
    validator.run()
