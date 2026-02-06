import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env 파일 로드
load_dotenv()

class LLMManager:
    """
    LM Studio(로컬) 또는 OpenAI 호환 서버에 연결하는 클래스입니다.
    """
    def __init__(self):
        self.base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        self.api_key = "lm-studio"  # 로컬 서버는 키가 필요 없지만 형식상 입력

    def get_model(self, model_name="local-model"):
        """
        ChatOpenAI 객체를 반환합니다. 
        LM Studio에서 로드한 모델의 이름이 다를 경우 model_name을 수정하세요.
        """
        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model_name,
            temperature=0  # KPI 답변은 정확해야 하므로 창의성을 낮춤(0)
        )

if __name__ == "__main__":
    # 연결 테스트
    manager = LLMManager()
    llm = manager.get_model()
    try:
        response = llm.invoke("안녕? 너는 어떤 일을 할 수 있어?")
        print("🤖 LLM 응답:", response.content)
    except Exception as e:
        print("❌ 연결 실패! LM Studio의 'Start Server' 버튼을 눌렀는지 확인하세요.")
        print("에러 내용:", e)