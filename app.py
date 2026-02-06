import streamlit as st
from agent import KPIAgent  # 우리가 만든 에이전트 클래스 이름에 맞춰주세요
import time

# 1. 페이지 설정
st.set_page_config(page_title="제조 KPI 인텔리전트 에이전트", page_icon="🏭")

# 2. 에이전트 초기화 (세션 상태 저장)
if "agent" not in st.session_state:
    with st.spinner("AI 에이전트의 뇌를 가동 중입니다..."):
        st.session_state.agent = KPIAgent()
        st.session_state.messages = []

# 3. 사이드바 (상태 표시)
with st.sidebar:
    st.title("🏭 공정 관리 센터")
    st.info("현재 LM Studio와 Supabase DB가 연결되어 있습니다.")
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 KPI 분석 비서")
st.caption("DB 수치 조회부터 매뉴얼 검색까지 한 번에 물어보세요.")

# 4. 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 사용자 입력 및 에이전트 실행
if prompt := st.chat_input("질문을 입력하세요 (예: 어제 불량률 얼마였어?)"):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 에이전트 답변 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("데이터 분석 중..."):
            # 에이전트 호출
            response = st.session_state.agent.invoke(prompt)
            
            # 1. response에서 텍스트 추출
            if isinstance(response, dict):
                raw_text = response.get('final_answer', str(response))
            else:
                raw_text = str(response)

            # 2. [수정] 정규표현식으로 content="내용" 내부의 텍스트만 추출
            import re
            # 따옴표 종류( " 또는 ' )에 상관없이 내부 텍스트를 가져옵니다.
            match = re.search(r'content=["\'](.*?)["\']', raw_text, re.DOTALL)
            
            if match:
                full_response = match.group(1)
                # 깨짐 방지: 이스케이프된 줄바꿈(\n)만 실제 줄바꿈으로 교체
                full_response = full_response.replace('\\n', '\n')
            else:
                # 패턴이 안 맞을 경우 지저분한 앞뒤 정보 제거
                full_response = raw_text.split("content=")[-1].split("additional_kwargs")[0].strip(" \"',")

            # 3. [추가] 불필요한 시스템 로그(query_date=None 등)가 섞여 있다면 제거
            if "DB 조회 결과" in full_response:
                # 사용자가 보기 좋게 데이터 부분만 깔끔하게 정리 (선택 사항)
                full_response = full_response.split("DB 조회 결과")[0].strip()

            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})