# router_graph.py
from __future__ import annotations

import json
import re
from typing import Literal, Optional, Tuple
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from llm import LLMManager  # 앞서 만든 llm.py의 LLMManager를 사용한다고 가정


Route = Literal["DB_QUERY", "RAG_SEARCH", "GENERAL_ANSWER"]


class RoutingState(TypedDict, total=False):
    question: str
    route: Route
    rationale: str
    answer: str


class QueryRouter:
    """
    규칙 기반 + LLM 기반(검증용/보조) 라우터입니다.
    최종 라우팅은 반드시 아래 규칙을 만족하도록 강제합니다.

    규칙(우선순위):
    1) 질문에 '생산량' 또는 '불량률'이 있고 숫자(또는 %)가 포함되면 DB_QUERY
    2) 질문에 '규정' 또는 '기준' 또는 '방법'이 포함되면 RAG_SEARCH
    3) 그 외 GENERAL_ANSWER
    """

    def __init__(self, llm_manager: Optional[LLMManager] = None) -> None:
        self.llm = llm_manager or LLMManager()

        self._kpi_keywords = ("생산량", "불량률")
        self._rag_keywords = ("규정", "기준", "방법")
        self._number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?%?")

    def classify(self, question: str) -> Tuple[Route, str]:
        if not question or not question.strip():
            return "GENERAL_ANSWER", "질문이 비어 있어 GENERAL_ANSWER로 분류했습니다."

        rule_route, rule_reason = self._classify_by_rule(question)

        llm_route, llm_reason = self._classify_by_llm(question, rule_route)

        # LLM 결과가 규칙과 다르면 규칙을 우선합니다(요구사항 준수 강제).
        if llm_route != rule_route:
            rationale = (
                f"{rule_reason} "
                f"(LLM 제안: {llm_route} / {llm_reason}) "
                f"규칙 우선으로 {rule_route}를 최종 선택했습니다."
            )
            return rule_route, rationale

        rationale = f"{rule_reason} (LLM 확인: {llm_reason})"
        return rule_route, rationale

    def _classify_by_rule(self, question: str) -> Tuple[Route, str]:
        has_kpi_kw = any(kw in question for kw in self._kpi_keywords)
        has_number = bool(self._number_pattern.search(question))
        if has_kpi_kw and has_number:
            return "DB_QUERY", "질문에 KPI 키워드(생산량/불량률)와 수치가 함께 포함되어 DB_QUERY로 분류했습니다."

        has_rag_kw = any(kw in question for kw in self._rag_keywords)
        if has_rag_kw:
            return "RAG_SEARCH", "질문에 규정/기준/방법 관련 키워드가 포함되어 RAG_SEARCH로 분류했습니다."

        return "GENERAL_ANSWER", "명시적 KPI 수치/규정 키워드가 없어 GENERAL_ANSWER로 분류했습니다."

    def _classify_by_llm(self, question: str, fallback: Route) -> Tuple[Route, str]:
        prompt = self._build_router_prompt(question)
        raw = self._ask_llm(prompt)

        route = self._safe_extract_route(raw)
        if route is None:
            return fallback, f"LLM 응답 파싱 실패로 fallback({fallback}) 적용. raw={raw!r}"

        if route not in ("DB_QUERY", "RAG_SEARCH", "GENERAL_ANSWER"):
            return fallback, f"LLM route 값이 유효하지 않아 fallback({fallback}) 적용. raw={raw!r}"

        reason = self._safe_extract_reason(raw) or "LLM이 규칙에 따라 분류했습니다."
        return route, reason

    @staticmethod
    def _build_router_prompt(question: str) -> str:
        # LLM이 “규칙대로만” 분류하게 하는 프롬프트입니다.
        return (
            "너는 라우터다. 아래 규칙을 반드시 그대로 적용해 분류해라.\n"
            "규칙(우선순위):\n"
            "1) 질문에 '생산량' 또는 '불량률'이 있고, 숫자(예: 1200, 15%, 0.12)가 포함되면 DB_QUERY\n"
            "2) 질문에 '규정' 또는 '기준' 또는 '방법'이 포함되면 RAG_SEARCH\n"
            "3) 그 외는 GENERAL_ANSWER\n\n"
            "반드시 다음 JSON 형태로만 답해라:\n"
            '{"route":"DB_QUERY|RAG_SEARCH|GENERAL_ANSWER","rationale":"한 문장 근거"}\n\n'
            f"질문: {question}"
        )

    def _ask_llm(self, prompt: str) -> str:
        """
        LLMManager의 인터페이스가 프로젝트마다 달라질 수 있어, 여러 방식으로 호출을 시도합니다.
        llm.py의 LLMManager가 아래 중 하나를 제공한다고 가정합니다.
        - invoke(prompt) -> str
        - chat(prompt) -> str
        - generate(prompt) -> str
        - __call__(prompt) -> str
        """
        for method_name in ("invoke", "chat", "generate"):
            if hasattr(self.llm, method_name):
                out = getattr(self.llm, method_name)(prompt)
                return out if isinstance(out, str) else str(out)
        if callable(self.llm):
            out = self.llm(prompt)
            return out if isinstance(out, str) else str(out)
        raise AttributeError("LLMManager에서 사용 가능한 호출 메서드를 찾지 못했습니다. (invoke/chat/generate/__call__)")

    @staticmethod
    def _safe_extract_route(raw: str) -> Optional[str]:
        # 1) JSON 파싱 시도
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "route" in obj:
                return str(obj["route"]).strip()
        except Exception:
            pass

        # 2) JSON이 깨진 경우 문자열에서 라벨 추출
        for label in ("DB_QUERY", "RAG_SEARCH", "GENERAL_ANSWER"):
            if label in raw:
                return label
        return None

    @staticmethod
    def _safe_extract_reason(raw: str) -> Optional[str]:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "rationale" in obj:
                return str(obj["rationale"]).strip()
        except Exception:
            return None
        return None


def build_router_graph(llm_manager: Optional[LLMManager] = None):
    """
    LangGraph StateGraph를 구성해 Router -> (DB/RAG/GENERAL)로 분기하는 그래프를 반환합니다.
    DB_QUERY, RAG_SEARCH, GENERAL_ANSWER 노드는 여기서는 예시로만 구현되어 있고,
    실제 프로젝트에서는 각 노드에서 DB 조회/RAG 검색/일반 답변 생성 로직을 붙이면 됩니다.
    """
    router = QueryRouter(llm_manager=llm_manager)

    graph = StateGraph(RoutingState)

    def route_node(state: RoutingState) -> RoutingState:
        q = state.get("question", "")
        route, rationale = router.classify(q)
        return {"question": q, "route": route, "rationale": rationale}

    def db_query_node(state: RoutingState) -> RoutingState:
        # 실제 구현에서는 여기서 Supabase 조회 등 DB_QUERY 로직을 수행하면 됩니다.
        return {**state, "answer": f"[DB_QUERY] 처리 대상입니다. rationale={state.get('rationale','')}"}

    def rag_search_node(state: RoutingState) -> RoutingState:
        # 실제 구현에서는 여기서 Chroma RAG 검색을 수행하면 됩니다.
        return {**state, "answer": f"[RAG_SEARCH] 처리 대상입니다. rationale={state.get('rationale','')}"}

    def general_answer_node(state: RoutingState) -> RoutingState:
        # 실제 구현에서는 일반 LLM 답변을 생성하면 됩니다.
        return {**state, "answer": f"[GENERAL_ANSWER] 처리 대상입니다. rationale={state.get('rationale','')}"}

    graph.add_node("router", route_node)
    graph.add_node("db_query", db_query_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("general_answer", general_answer_node)

    graph.add_edge(START, "router")

    # Router 결과에 따라 분기
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "DB_QUERY": "db_query",
            "RAG_SEARCH": "rag_search",
            "GENERAL_ANSWER": "general_answer",
        },
    )

    # 각 처리 후 종료
    graph.add_edge("db_query", END)
    graph.add_edge("rag_search", END)
    graph.add_edge("general_answer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_router_graph()

    tests = [
        "생산량 1200일 때 불량률 0.08이면 이상치야?",
        "불량률 기준이 뭐야? 규정에 따르면 어떻게 해석해?",
        "오늘 공장 운영 효율을 올리는 아이디어가 있을까?",
    ]

    for q in tests:
        out = app.invoke({"question": q})
        print("\nQ:", q)
        print("route:", out.get("route"))
        print("answer:", out.get("answer"))
        print("rationale:", out.get("rationale"))
