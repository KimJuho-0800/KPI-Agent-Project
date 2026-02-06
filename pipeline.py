from main import generate_manufacturing_kpi_csv
from validator import ManufacturingKPIValidator
from database import SupabaseDatabase

def run_pipeline():
    print("🚀 1단계: 가상 KPI 데이터 생성 중...")
    generate_manufacturing_kpi_csv()

    print("🔍 2단계: 데이터 유효성 검사 중...")
    validator = ManufacturingKPIValidator()
    result = validator.run()

    if result.prod_qty_invalid_count == 0 and result.defect_rate_outlier_count == 0:
        print("✅ 검증 완료! 데이터를 Supabase에 업로드합니다.")
        
        # 3단계: DB 업로드
        db = SupabaseDatabase()
        df = validator.load() # 검증된 데이터 로드
        inserted_count = db.insert_kpi_dataframe(df)
        
        print(f"🎉 성공! 총 {inserted_count}개의 데이터가 클라우드 DB에 저장되었습니다.")
    else:
        print("❌ 데이터에 이상이 있어 업로드를 중단합니다. 리포트를 확인하세요.")

if __name__ == "__main__":
    run_pipeline()