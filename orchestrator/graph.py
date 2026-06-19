"""LangGraph orchestrator for multi-agent system."""
from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    ArchitectAgent,
    CodeStyleReviewer,
    CodingAgent,
    ComposerReviewAgent,
    CorrectnessReviewer,
    ResponseAggregator,
    SecurityReviewer,
    TesterAgent,
)
from agents.review_models import ReviewVerdict
from agents.review_utils import build_review_context
from config import settings
from config.settings import TesterMode

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, dict[str, Any]], None]


class AgentState(TypedDict):
    """State shared across all agents."""
    query: str
    agent_response: dict
    all_responses: Annotated[list, operator.add]
    final_response: str
    verified: bool
    composer_verdict: dict
    refinement_feedback: dict
    iteration: int
    max_iterations: int
    workspace_path: str
    tester_mode: str
    test_command: str | None
    architecture_output: dict
    test_authoring_output: dict
    coding_output: dict
    testing_output: dict
    correctness_review: dict
    security_review: dict
    code_style_review: dict
    review_context: str
    pipeline_trace: Annotated[list, operator.add]


class LangGraphOrchestrator:
    """Main orchestrator using LangGraph."""

    def __init__(self):
        """Initialize orchestrator with all agents."""
        self.architect_agent = ArchitectAgent()
        self.coding_agent = CodingAgent()
        self.tester_agent = TesterAgent()
        self.correctness_reviewer = CorrectnessReviewer()
        self.security_reviewer = SecurityReviewer()
        self.code_style_reviewer = CodeStyleReviewer()
        self.composer = ComposerReviewAgent()
        self.aggregator = ResponseAggregator()
        self._progress_callback: ProgressCallback | None = None

        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _wrap_node(self, stage: str, fn: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
        """Add logging and progress reporting around a pipeline node."""

        def wrapped(state: AgentState) -> AgentState:
            iteration = state.get("iteration", 0)
            query_preview = state.get("query", "")[:100]
            logger.info(
                "[pipeline] START stage=%s iteration=%d query=%r",
                stage,
                iteration,
                query_preview,
            )
            if self._progress_callback:
                self._progress_callback("start", stage, dict(state))

            try:
                updates = fn(state)
                trace_entries = updates.get("pipeline_trace", [])
                last_trace = trace_entries[-1] if trace_entries else {"stage": stage}
                logger.info("[pipeline] DONE stage=%s iteration=%d", stage, iteration)
                if self._progress_callback:
                    self._progress_callback(
                        "done",
                        stage,
                        {**dict(state), **updates, "_last_trace": last_trace},
                    )
                return updates
            except Exception as exc:
                logger.exception(
                    "[pipeline] FAILED stage=%s iteration=%d error=%s",
                    stage,
                    iteration,
                    exc,
                )
                if self._progress_callback:
                    self._progress_callback(
                        "failed",
                        stage,
                        {**dict(state), "_error": str(exc)},
                    )
                raise

        return wrapped

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        workflow.add_node("architect", self._wrap_node("architect", self._architect_node))
        workflow.add_node("write_tests", self._wrap_node("write_tests", self._write_tests_node))
        workflow.add_node("coding", self._wrap_node("coding", self._coding_node))
        workflow.add_node("run_tests", self._wrap_node("run_tests", self._run_tests_node))
        workflow.add_node(
            "prepare_review_context",
            self._wrap_node("prepare_review_context", self._prepare_review_context_node),
        )
        workflow.add_node(
            "correctness_reviewer",
            self._wrap_node("correctness_reviewer", self._correctness_reviewer_node),
        )
        workflow.add_node(
            "security_reviewer",
            self._wrap_node("security_reviewer", self._security_reviewer_node),
        )
        workflow.add_node(
            "code_style_reviewer",
            self._wrap_node("code_style_reviewer", self._code_style_reviewer_node),
        )
        workflow.add_node("composer", self._wrap_node("composer", self._composer_node))
        workflow.add_node("aggregate", self._wrap_node("aggregate", self._aggregate_node))

        workflow.set_entry_point("architect")
        workflow.add_edge("architect", "coding")
        workflow.add_conditional_edges(
            "coding",
            self._route_after_coding,
            {
                "write_tests": "write_tests",
                "prepare_review_context": "prepare_review_context",
            },
        )
        workflow.add_edge("write_tests", "run_tests")
        workflow.add_edge("run_tests", "prepare_review_context")
        workflow.add_edge("prepare_review_context", "correctness_reviewer")
        workflow.add_edge("correctness_reviewer", "security_reviewer")
        workflow.add_edge("security_reviewer", "code_style_reviewer")
        workflow.add_edge("code_style_reviewer", "composer")
        workflow.add_conditional_edges(
            "composer",
            self._route_after_composer,
            {
                "coding": "coding",
                "aggregate": "aggregate",
            },
        )
        workflow.add_edge("aggregate", END)

        return workflow

    def _route_after_composer(self, state: AgentState) -> Literal["coding", "aggregate"]:
        verdict = state.get("composer_verdict", {})
        next_stage = "aggregate"
        if verdict.get("verdict") == ReviewVerdict.REFINE.value:
            if state.get("iteration", 0) < state.get("max_iterations", settings.max_iterations):
                next_stage = "coding"
        logger.info(
            "[pipeline] ROUTE after composer -> %s (verdict=%s iteration=%d)",
            next_stage,
            verdict.get("verdict"),
            state.get("iteration", 0),
        )
        return next_stage  # type: ignore[return-value]

    @staticmethod
    def _resolve_tester_mode(state: AgentState) -> TesterMode:
        raw = state.get("tester_mode") or settings.tester_mode
        if isinstance(raw, TesterMode):
            return raw
        return TesterMode(raw)

    def _route_after_coding(self, state: AgentState) -> Literal["write_tests", "prepare_review_context"]:
        mode = self._resolve_tester_mode(state)
        if mode == TesterMode.BENCHMARKING:
            next_stage = "prepare_review_context"
            logger.info(
                "[pipeline] ROUTE after coding -> %s (tester_mode=%s, run_tests skipped)",
                next_stage,
                mode.value,
            )
        else:
            next_stage = "write_tests"
            logger.info("[pipeline] ROUTE after coding -> %s (tester_mode=%s)", next_stage, mode.value)
        return next_stage  # type: ignore[return-value]

    def _pipeline_context(self, state: AgentState) -> tuple[str, dict, dict, dict, dict, str]:
        return (
            state["query"],
            state.get("architecture_output", {}),
            state.get("test_authoring_output", {}),
            state.get("coding_output", {}),
            state.get("testing_output", {}),
            state.get("workspace_path", settings.workspace_path),
        )

    def _architect_node(self, state: AgentState) -> AgentState:
        response = self.architect_agent.design(
            query=state["query"],
            workspace_path=state.get("workspace_path", settings.workspace_path),
        )
        return {
            "architecture_output": response,
            "agent_response": response,
            "all_responses": [response],
            "pipeline_trace": [
                {
                    "stage": "architect",
                    "result": response.get("response", ""),
                }
            ],
        }

    def _write_tests_node(self, state: AgentState) -> AgentState:
        response = self.tester_agent.write_tests(
            query=state["query"],
            architecture_output=state.get("architecture_output", {}),
            coding_output=state.get("coding_output", {}),
            workspace_path=state.get("workspace_path", settings.workspace_path),
        )
        return {
            "test_authoring_output": response,
            "agent_response": response,
            "all_responses": [response],
            "pipeline_trace": [
                {
                    "stage": "write_tests",
                    "result": response.get("response", ""),
                    "sdk_status": response.get("sdk_status", "unknown"),
                    "sdk_run_id": response.get("sdk_run_id"),
                }
            ],
        }

    def _coding_node(self, state: AgentState) -> AgentState:
        refinement_feedback = state.get("refinement_feedback") or None
        response = self.coding_agent.execute(
            query=state["query"],
            architecture_output=state.get("architecture_output", {}),
            workspace_path=state.get("workspace_path", settings.workspace_path),
            refinement_feedback=refinement_feedback,
        )
        trace_stage = "coding_refinement" if refinement_feedback else "coding"
        return {
            "coding_output": response,
            "agent_response": response,
            "all_responses": [response],
            "pipeline_trace": [
                {
                    "stage": trace_stage,
                    "iteration": state.get("iteration", 0),
                    "result": response.get("response", ""),
                    "sdk_status": response.get("sdk_status", "unknown"),
                    "sdk_run_id": response.get("sdk_run_id"),
                }
            ],
        }

    def _run_tests_node(self, state: AgentState) -> AgentState:
        response = self.tester_agent.run_tests(
            query=state["query"],
            coding_output=state.get("coding_output", {}),
            test_authoring_output=state.get("test_authoring_output", {}),
            workspace_path=state.get("workspace_path", settings.workspace_path),
            tester_mode=self._resolve_tester_mode(state).value,
            test_command=state.get("test_command"),
        )
        return {
            "testing_output": response,
            "agent_response": response,
            "all_responses": [response],
            "pipeline_trace": [
                {
                    "stage": "run_tests",
                    "iteration": state.get("iteration", 0),
                    "result": response.get("response", ""),
                    "passed": response.get("passed", False),
                    "sdk_status": response.get("sdk_status", "unknown"),
                    "sdk_run_id": response.get("sdk_run_id"),
                }
            ],
        }

    def _prepare_review_context_node(self, state: AgentState) -> AgentState:
        query, architecture, test_authoring, coding_output, testing_output, workspace_path = self._pipeline_context(state)
        tester_mode = self._resolve_tester_mode(state)
        benchmarking = tester_mode == TesterMode.BENCHMARKING
        context = build_review_context(
            query,
            coding_output,
            testing_output,
            workspace_path,
            test_authoring_output=test_authoring,
            architecture_output=architecture,
            benchmarking=benchmarking,
            tester_mode=tester_mode.value,
        )
        logger.info(
            "[pipeline] review context built once (%d chars) for iteration=%d",
            len(context),
            state.get("iteration", 0),
        )
        return {
            "review_context": context,
            "pipeline_trace": [{"stage": "prepare_review_context", "context_chars": len(context)}],
        }

    def _correctness_reviewer_node(self, state: AgentState) -> AgentState:
        query, architecture, test_authoring, coding_output, testing_output, workspace_path = self._pipeline_context(state)
        review = self.correctness_reviewer.review(
            query=query,
            coding_output=coding_output,
            test_authoring_output=test_authoring,
            testing_output=testing_output,
            workspace_path=workspace_path,
            tester_mode=self._resolve_tester_mode(state).value,
            architecture_output=architecture,
            review_context=state.get("review_context", ""),
        )
        payload = review.model_dump(mode="json")
        return {
            "correctness_review": payload,
            "pipeline_trace": [{"stage": "correctness_reviewer", "result": payload}],
        }

    def _security_reviewer_node(self, state: AgentState) -> AgentState:
        query, architecture, test_authoring, coding_output, testing_output, workspace_path = self._pipeline_context(state)
        review = self.security_reviewer.review(
            query=query,
            coding_output=coding_output,
            testing_output=testing_output,
            workspace_path=workspace_path,
            test_authoring_output=test_authoring,
            tester_mode=self._resolve_tester_mode(state).value,
            architecture_output=architecture,
            review_context=state.get("review_context", ""),
        )
        payload = review.model_dump(mode="json")
        return {
            "security_review": payload,
            "pipeline_trace": [{"stage": "security_reviewer", "result": payload}],
        }

    def _code_style_reviewer_node(self, state: AgentState) -> AgentState:
        query, architecture, test_authoring, coding_output, testing_output, workspace_path = self._pipeline_context(state)
        review = self.code_style_reviewer.review(
            query=query,
            coding_output=coding_output,
            testing_output=testing_output,
            workspace_path=workspace_path,
            test_authoring_output=test_authoring,
            tester_mode=self._resolve_tester_mode(state).value,
            architecture_output=architecture,
            review_context=state.get("review_context", ""),
        )
        payload = review.model_dump(mode="json")
        return {
            "code_style_review": payload,
            "pipeline_trace": [{"stage": "code_style_reviewer", "result": payload}],
        }

    def _composer_node(self, state: AgentState) -> AgentState:
        from agents.review_models import ReviewerResult

        correctness = ReviewerResult.model_validate(state.get("correctness_review", {}))
        security = ReviewerResult.model_validate(state.get("security_review", {}))
        code_style = ReviewerResult.model_validate(state.get("code_style_review", {}))

        verdict = self.composer.compose(
            query=state["query"],
            correctness=correctness,
            security=security,
            code_style=code_style,
        )
        payload = verdict.model_dump(mode="json")
        updates: dict[str, Any] = {
            "composer_verdict": payload,
            "verified": verdict.verdict == ReviewVerdict.PASS,
            "agent_response": {
                "agent": "composer",
                "response": verdict.summary,
                "verdict": payload,
            },
            "pipeline_trace": [{"stage": "composer", "result": payload}],
        }
        if verdict.verdict == ReviewVerdict.REFINE:
            updates["refinement_feedback"] = self.composer.to_feedback_dict(verdict)
            updates["iteration"] = state.get("iteration", 0) + 1
        return updates

    def _aggregate_node(self, state: AgentState) -> AgentState:
        all_responses = state.get("all_responses", [])
        if not all_responses:
            all_responses = [state.get("agent_response", {})]

        aggregated = self.aggregator.aggregate_responses(all_responses, state["query"])
        composer_verdict = state.get("composer_verdict", {})
        if composer_verdict:
            aggregated = {**aggregated, "composer_verdict": composer_verdict}
        return {
            "final_response": aggregated.get("response", ""),
            "agent_response": aggregated,
        }

    def invoke(
        self,
        query: str,
        workspace_path: str | None = None,
        max_iterations: int | None = None,
        tester_mode: str | TesterMode | None = None,
        test_command: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        iteration_limit = max_iterations if max_iterations is not None else settings.max_iterations
        resolved_workspace = workspace_path or settings.workspace_path
        resolved_mode = tester_mode or settings.tester_mode
        if isinstance(resolved_mode, TesterMode):
            resolved_mode = resolved_mode.value
        initial_state = {
            "query": query,
            "agent_response": {},
            "all_responses": [],
            "final_response": "",
            "verified": False,
            "composer_verdict": {},
            "refinement_feedback": {},
            "iteration": 0,
            "max_iterations": iteration_limit,
            "workspace_path": resolved_workspace,
            "tester_mode": resolved_mode,
            "test_command": test_command,
            "architecture_output": {},
            "test_authoring_output": {},
            "coding_output": {},
            "testing_output": {},
            "correctness_review": {},
            "security_review": {},
            "code_style_review": {},
            "review_context": "",
            "pipeline_trace": [],
        }

        logger.info(
            "[pipeline] INVOKE query=%r max_iterations=%d workspace=%s tester_mode=%s test_command=%r",
            query[:100],
            iteration_limit,
            resolved_workspace,
            resolved_mode,
            test_command,
        )
        self._progress_callback = progress_callback
        try:
            return self.app.invoke(initial_state)
        finally:
            self._progress_callback = None
