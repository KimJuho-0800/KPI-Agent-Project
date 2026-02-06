# agent.py
# Python 3.11 / LangGraph 0.3.x / LangChain 사용 전제
#
# 목적:
# - 사용자의 질문을 받으면 Router(의도 분류) -> Tools(DB/RAG 조회) -> Generator(최종 답변 생성) 흐름으로 동작하는 에이전트.
# - database.py(SupabaseDatabase), rag_storage.py(KPIMannualRAGStorage), llm.py(LLMManager)를 조립해 완성.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Literal, Optional, Tuple
from typing_extensions import TypedDict

import pandas as pd

from langgraph.graph import StateGraph, START, END

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# 사용자가 이미 만들어둔 부품들(파일/클래스명은 질문에서 지정한 그대로 import)
from database import SupabaseDatabase
from rag_storage import KPIMannualRAGStorage
from llm import LLMManager


Route = Literal["DB_QUERY", "RAG_SEARCH", "GENERAL_ANSWER"]


class AgentState(TypedDict, total=False):
    """
    LangGraph 상태 정의.
    - messages: 대화 메시지 히스토리(최소 1개의 HumanMessage를 포함)
    - context: Router/Tools 단계에서 수집한 정보를 담는 컨테이너
    """
    messages: list[BaseMessage]
    context: Dict[str, Any]

    # 내부 동작에 필요한 확장 필드(필수는 아니지만 total=False로 유연하게 사용)
    route: Route
    rationale: str
    tool_result: Dict[str, Any]
    final_answer: str


@dataclass(frozen=True)
class RouterDecision:
    route: Route
    rationale: str


class KPIAgent:
    """
    - Router: LLM(우선) + 규칙 기반(fallback)으로 의도 분류
    - Tools: DB 또는 RAG 호출로 필요한 정보 수집
    - Generator: 수집된 정보를 포함해 LLMManager(LM Studio)로 최종 답변 생성
    """

    def __init__(
        self,
        db: Optional[SupabaseDatabase] = None,
        knowledge: Optional[KPIMannualRAGStorage] = None,
        llm: Optional[LLMManager] = None,
        supabase_table: str = "kpi_data",
        rag_top_k: int = 4,
    ) -> None:
        self.db = db or SupabaseDatabase()
        self.knowledge = knowledge or KPIMannualRAGStorage()
        # LLMManager 인스턴스를 만든 후, .get_model()을 호출해서 실제 모델 객체를 할당합니다.
        llm_manager = LLMManager()
        self.llm = llm or llm_manager.get_model()

        self.supabase_table = supabase_table
        self.rag_top_k = int(rag_top_k)

        # 라우팅 규칙에 사용할 키워드/패턴
        self._kpi_keywords = ("생산량", "불량률")
        self._rag_keywords = ("규정", "대응 방법", "기준", "방법")
        self._number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?%?")  # 1200, 0.12, 15% 등
        self._date_pattern = re.compile(
            r"(?P<y>\d{4})[./-](?P<m>\d{1,2})[./-](?P<d>\d{1,2})"
        )  # 2026-02-01 / 2026.2.1 / 2026/02/01

        self.app = self._build_graph()

    # -------------------------
    # Graph 구성
    # -------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("router", self._node_router)
        graph.add_node("tools", self._node_tools)
        graph.add_node("generator", self._node_generator)

        # 요구한 3단계 흐름을 직렬로 연결
        graph.add_edge(START, "router")
        graph.add_edge("router", "tools")
        graph.add_edge("tools", "generator")
        graph.add_edge("generator", END)

        return graph.compile()

    def invoke(self, question: str) -> AgentState:
        """
        외부에서 한 번 호출하면 에이전트가 3단계를 수행하고 최종 답을 messages에 추가해 반환합니다.
        """
        init_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "context": {},
        }
        return self.app.invoke(init_state)

    # -------------------------
    # Node 1: Router
    # -------------------------
    def _node_router(self, state: AgentState) -> AgentState:
        question = self._extract_latest_user_question(state)
        decision = self._route(question)

        ctx = dict(state.get("context", {}))
        ctx["router"] = {"route": decision.route, "rationale": decision.rationale}

        return {
            **state,
            "route": decision.route,
            "rationale": decision.rationale,
            "context": ctx,
        }

    def _route(self, question: str) -> RouterDecision:
        """
        1) LLM에게 규칙에 맞춘 분류를 요청
        2) 실패/애매하면 규칙 기반 fallback
        3) 최종 결과는 반드시 규칙을 만족하도록 강제(요구사항 준수)
        """
        rule_route, rule_reason = self._route_by_rule(question)

        llm_route, llm_reason = self._route_by_llm(question)
        if llm_route is None:
            return RouterDecision(route=rule_route, rationale=f"{rule_reason} (LLM 파싱 실패로 규칙 적용)")

        # LLM이 규칙과 다르게 말하면, 요구사항 상 규칙 우선으로 강제
        if llm_route != rule_route:
            rationale = (
                f"{rule_reason} (LLM 제안: {llm_route} / {llm_reason}) "
                f"규칙 우선으로 {rule_route}로 확정했습니다."
            )
            return RouterDecision(route=rule_route, rationale=rationale)

        return RouterDecision(route=rule_route, rationale=f"{rule_reason} (LLM 확인: {llm_reason})")

    def _route_by_rule(self, question: str) -> Tuple[Route, str]:
        """
        요구사항의 분류 규칙(우선순위)을 그대로 구현:
        - 질문에 '생산량' 또는 '불량률'이 있고 숫자(또는 %)가 있으면 DB_QUERY
        - 질문에 '규정', '기준', '방법'(또는 '대응 방법')이 있으면 RAG_SEARCH
        - 나머지 GENERAL_ANSWER
        """
        has_kpi_kw = any(k in question for k in self._kpi_keywords)
        has_number = bool(self._number_pattern.search(question))
        if has_kpi_kw and has_number:
            return "DB_QUERY", "질문에 KPI 키워드(생산량/불량률)와 수치 표현이 함께 있어 DB_QUERY로 분류했습니다."

        has_rag_kw = any(k in question for k in self._rag_keywords)
        if has_rag_kw:
            return "RAG_SEARCH", "질문에 규정/기준/방법 관련 키워드가 있어 RAG_SEARCH로 분류했습니다."

        return "GENERAL_ANSWER", "KPI 수치/규정 키워드가 명확하지 않아 GENERAL_ANSWER로 분류했습니다."

    def _route_by_llm(self, question: str) -> Tuple[Optional[Route], str]:
        """
        LLMManager에게 '규칙 기반 분류'를 요청하고 JSON 응답을 파싱합니다.
        """
        prompt = (
            "너는 라우터다. 아래 규칙을 반드시 그대로 적용해 route를 결정해라.\n"
            "규칙(우선순위):\n"
            "1) 질문에 '생산량' 또는 '불량률'이 있고 숫자(예: 1200, 15%, 0.12)가 포함되면 DB_QUERY\n"
            "2) 질문에 '규정', '기준', '방법', '대응 방법' 같은 단어가 포함되면 RAG_SEARCH\n"
            "3) 그 외 GENERAL_ANSWER\n\n"
            "반드시 다음 JSON 형태로만 답해라:\n"
            '{"route":"DB_QUERY|RAG_SEARCH|GENERAL_ANSWER","rationale":"한 문장 근거"}\n\n'
            f"질문: {question}"
        )

        raw = self._call_llm(prompt)
        route = self._safe_extract_route(raw)
        rationale = self._safe_extract_rationale(raw) or raw.strip()

        if route not in ("DB_QUERY", "RAG_SEARCH", "GENERAL_ANSWER"):
            return None, f"LLM 응답에서 route 파싱 실패: raw={raw!r}"

        return route, rationale

    # -------------------------
    # Node 2: Tools
    # -------------------------
    def _node_tools(self, state: AgentState) -> AgentState:
        question = self._extract_latest_user_question(state)
        route: Route = state.get("route", "GENERAL_ANSWER")

        tool_result: Dict[str, Any] = {}
        ctx = dict(state.get("context", {}))

        # route에 따라 필요한 도구 실행
        if route == "DB_QUERY":
            tool_result = self._tool_db_query(question)
            ctx["db"] = tool_result
        elif route == "RAG_SEARCH":
            tool_result = self._tool_rag_search(question)
            ctx["rag"] = tool_result
        else:
            tool_result = {"note": "GENERAL_ANSWER: 외부 조회 없이 답변 생성"}
            ctx["tools"] = tool_result

        return {**state, "tool_result": tool_result, "context": ctx}

    def _tool_db_query(self, question: str) -> Dict[str, Any]:
        """
        SupabaseDatabase를 사용해 KPI 숫자를 가져옵니다.

        구현 전략:
        - 질문에 날짜(YYYY-MM-DD 등)가 있으면 해당 date의 row 조회
        - 없으면 가장 최근 row 1개 조회
        - SupabaseDatabase가 client를 노출하는 경우(이전 구현 포함) 직접 select 수행
        - 프로젝트마다 DB wrapper가 다를 수 있으니, 가능한 메서드가 있으면 우선 호출 후 fallback
        """
        # 1) 사용자 질문에서 날짜 추출(있으면 그 날짜 기준 조회)
        target_date = self._extract_date(question)

        # 2) DB wrapper가 특정 메서드를 제공한다면 우선 사용
        #    예: db.query_kpi(date=...), db.fetch_kpi(...), db.get_latest_kpi()
        #    (프로젝트 구현에 맞게 자동으로 탐색)
        candidates = []
        if target_date:
            candidates.extend([
                ("query_kpi_by_date", {"target_date": target_date}),
                ("get_kpi_by_date", {"target_date": target_date}),
                ("fetch_by_date", {"target_date": target_date}),
            ])
        candidates.extend([
            ("get_latest_kpi", {}),
            ("fetch_latest_kpi", {}),
            ("query_latest_kpi", {}),
        ])

        for name, kwargs in candidates:
            if hasattr(self.db, name):
                fn = getattr(self.db, name)
                try:
                    data = fn(**kwargs)  # type: ignore[misc]
                    return {
                        "source": f"SupabaseDatabase.{name}",
                        "query_date": str(target_date) if target_date else None,
                        "rows": data,
                    }
                except Exception as e:
                    return {
                        "source": f"SupabaseDatabase.{name}",
                        "query_date": str(target_date) if target_date else None,
                        "error": str(e),
                    }

        # 3) fallback: SupabaseDatabase 내부 client로 직접 조회(이전 예시 database.py 구조와 호환)
        if not hasattr(self.db, "client"):
            return {
                "source": "SupabaseDatabase",
                "query_date": str(target_date) if target_date else None,
                "error": "SupabaseDatabase에 client가 없어 조회를 수행할 수 없습니다. DB 조회 메서드를 추가하세요.",
            }

        client = getattr(self.db, "client")

        try:
            q = client.table(self.supabase_table).select("date, production_qty, defect_rate")
            if target_date:
                q = q.eq("date", str(target_date))
                resp = q.execute()
            else:
                # 최신 데이터 1건
                resp = q.order("date", desc=True).limit(1).execute()

            # supabase-py 응답 형태는 버전에 따라 resp.data 혹은 dict일 수 있어 안전하게 처리
            data = getattr(resp, "data", None)
            if data is None and isinstance(resp, dict):
                data = resp.get("data")

            return {
                "source": f"SupabaseDatabase.client.table('{self.supabase_table}')",
                "query_date": str(target_date) if target_date else None,
                "rows": data or [],
            }
        except Exception as e:
            return {
                "source": f"SupabaseDatabase.client.table('{self.supabase_table}')",
                "query_date": str(target_date) if target_date else None,
                "error": str(e),
            }

    def _tool_rag_search(self, question: str) -> Dict[str, Any]:
        """
        KPIMannualRAGStorage를 사용해 PDF 매뉴얼에서 관련 지침/기준/방법을 검색합니다.
        """
        try:
            # KPIMannualRAGStorage 구현이 get_relevant_context를 제공하면 가장 간편
            if hasattr(self.knowledge, "get_relevant_context"):
                ctx = self.knowledge.get_relevant_context(question, k=self.rag_top_k)  # type: ignore[misc]
                return {"source": "KPIMannualRAGStorage.get_relevant_context", "context": ctx}

            # 그렇지 않으면 search 결과를 합쳐서 context 구성
            if hasattr(self.knowledge, "search"):
                hits = self.knowledge.search(question, k=self.rag_top_k)  # type: ignore[misc]
                # hits가 객체 리스트/문자열 리스트 등일 수 있으니 최대한 안전하게 문자열화
                text_parts = []
                for h in hits:
                    if isinstance(h, str):
                        text_parts.append(h)
                    elif hasattr(h, "content"):
                        text_parts.append(str(getattr(h, "content")))
                    else:
                        text_parts.append(str(h))
                return {"source": "KPIMannualRAGStorage.search", "context": "\n\n".join(text_parts), "hits": hits}

            return {"source": "KPIMannualRAGStorage", "error": "KPIMannualRAGStorage에 search/get_relevant_context가 없습니다."}
        except Exception as e:
            return {"source": "KPIMannualRAGStorage", "error": str(e)}

    # -------------------------
    # Node 3: Generator
    # -------------------------
    def _node_generator(self, state: AgentState) -> AgentState:
        question = self._extract_latest_user_question(state)
        route: Route = state.get("route", "GENERAL_ANSWER")
        tool_result = state.get("tool_result", {})
        ctx = state.get("context", {})

        # Generator 프롬프트 구성: 도구 결과를 "근거 컨텍스트"로 넣고 최종 답변 생성
        system = (
            "너는 제조 KPI 도우미다. 사용자의 질문에 한국어로 정확하고 간결하게 답해라. "
            "DB 결과가 있으면 수치와 날짜를 명확히 언급하고, "
            "매뉴얼(RAG) 결과가 있으면 해당 지침/기준/방법을 요약해 적용 방법까지 설명해라. "
            "근거가 부족하면 모른다고 말하고, 필요한 추가 정보(예: 날짜, 설비, 라인)를 짧게 요청해라."
        )

        context_block = self._format_context_for_generation(route=route, tool_result=tool_result, context=ctx)
        user_prompt = (
            f"질문: {question}\n\n"
            f"조회/검색 결과:\n{context_block}\n\n"
            "위 정보를 바탕으로 최종 답변을 작성해라."
        )

        answer = self._call_llm_with_messages(
            [
                SystemMessage(content=system),
                HumanMessage(content=user_prompt),
            ]
        )

        # messages에 AI 응답 추가
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=answer))

        ctx2 = dict(ctx)
        ctx2["final"] = {"route": route, "answer": answer}

        return {**state, "messages": messages, "final_answer": answer, "context": ctx2}

    # -------------------------
    # LLM 호출 유틸
    # -------------------------
    def _call_llm(self, prompt: str) -> str:
        """
        LLMManager의 실제 인터페이스가 프로젝트마다 다를 수 있어 여러 호출 방식을 순차 시도합니다.
        - invoke(prompt) / chat(prompt) / generate(prompt) / complete(prompt) / __call__(prompt)
        """
        for method_name in ("invoke", "chat", "generate", "complete"):
            if hasattr(self.llm, method_name):
                out = getattr(self.llm, method_name)(prompt)
                return out if isinstance(out, str) else str(out)

        if callable(self.llm):
            out = self.llm(prompt)
            return out if isinstance(out, str) else str(out)

        raise AttributeError("LLMManager에서 호출 가능한 메서드를 찾지 못했습니다. (invoke/chat/generate/complete/__call__)")

    def _call_llm_with_messages(self, messages: list[BaseMessage]) -> str:
        """
        LLMManager가 messages 기반 인터페이스를 제공하는 경우를 지원합니다.
        - invoke_messages(messages) / chat_messages(messages) 등
        없으면 messages 내용을 합쳐 단일 prompt로 호출합니다.
        """
        for method_name in ("invoke_messages", "chat_messages"):
            if hasattr(self.llm, method_name):
                out = getattr(self.llm, method_name)(messages)
                return out if isinstance(out, str) else str(out)

        # fallback: messages를 텍스트로 직렬화해서 단일 prompt로 호출
        merged = []
        for m in messages:
            role = "SYSTEM" if isinstance(m, SystemMessage) else "USER" if isinstance(m, HumanMessage) else "ASSISTANT"
            merged.append(f"[{role}]\n{m.content}")
        return self._call_llm("\n\n".join(merged))

    # -------------------------
    # 기타 유틸
    # -------------------------
    @staticmethod
    def _extract_latest_user_question(state: AgentState) -> str:
        messages = state.get("messages", [])
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return str(m.content)
        return ""

    def _extract_date(self, text: str) -> Optional[date]:
        """
        질문에 '2026-02-01' 같은 날짜가 있으면 date로 변환해 반환합니다.
        """
        m = self._date_pattern.search(text)
        if not m:
            return None
        y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
        try:
            return date(y, mo, d)
        except Exception:
            return None

    @staticmethod
    def _safe_extract_route(raw: str) -> Optional[Route]:
        """
        LLM 라우터 응답에서 route를 안전하게 추출합니다.
        """
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "route" in obj:
                return str(obj["route"]).strip()  # type: ignore[return-value]
        except Exception:
            pass

        # JSON이 깨졌을 경우 문자열에서 라벨 탐색
        for label in ("DB_QUERY", "RAG_SEARCH", "GENERAL_ANSWER"):
            if label in raw:
                return label  # type: ignore[return-value]
        return None

    @staticmethod
    def _safe_extract_rationale(raw: str) -> Optional[str]:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "rationale" in obj:
                return str(obj["rationale"]).strip()
        except Exception:
            return None
        return None

    @staticmethod
    def _format_context_for_generation(route: Route, tool_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Generator에 넣을 컨텍스트 문자열을 보기 좋게 구성합니다.
        """
        if route == "DB_QUERY":
            rows = tool_result.get("rows", [])
            qd = tool_result.get("query_date")
            if tool_result.get("error"):
                return f"DB 조회 중 오류: {tool_result.get('error')}"
            return f"DB 조회 결과(query_date={qd}): {rows}"
        if route == "RAG_SEARCH":
            if tool_result.get("error"):
                return f"RAG 검색 중 오류: {tool_result.get('error')}"
            return f"매뉴얼 검색 결과:\n{tool_result.get('context', '')}"

        return "외부 조회 없음"

# -------------------------
# 실행 예시
# -------------------------
if __name__ == "__main__":
    agent = KPIAgent()

    # 예시 1) KPI 숫자(수치) 포함 + 생산량/불량률 키워드 -> DB_QUERY
    out1 = agent.invoke("2026-02-01 생산량 1200이 맞는지 확인해줘. 불량률도 같이 알려줘.")
    print(out1.get("final_answer"))

    # 예시 2) 규정/기준/방법 -> RAG_SEARCH
    out2 = agent.invoke("불량률 이상치 기준은 매뉴얼 규정에 따라 어떻게 정의돼? 대응 방법도 알려줘.")
    print(out2.get("final_answer"))

    # 예시 3) 그 외 -> GENERAL_ANSWER
    out3 = agent.invoke("생산 효율을 올리기 위한 일반적인 개선 아이디어가 있을까?")
    print(out3.get("final_answer"))
