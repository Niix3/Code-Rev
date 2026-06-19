"""Final response aggregation for pipeline output."""
from typing import Any

from openai import OpenAI

from config import settings


class ResponseAggregator:
    """Synthesizes stage outputs into a user-facing final answer."""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def aggregate_responses(
        self,
        responses: list[dict[str, Any]],
        query: str,
    ) -> dict[str, Any]:
        if not responses:
            return {"response": "No responses to aggregate", "sources": []}

        if len(responses) == 1:
            return responses[0]

        combined = "\n\n".join(
            f"[{item.get('agent', 'unknown')}]: {item.get('response', '')}"
            for item in responses
        )

        system_prompt = """You are synthesizing multiple agent responses into a coherent final answer.
Combine the best information from each response, resolve conflicts, and provide
a unified, comprehensive answer."""
        user_prompt = f"Query: {query}\n\nAgent Responses:\n{combined}\n\nSynthesized Answer:"
        final_response = self.client.chat.completions.create(
            model=settings.default_llm_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        all_sources: list[str] = []
        for item in responses:
            if "sources" in item:
                all_sources.extend(item["sources"])

        return {
            "response": final_response.choices[0].message.content or "",
            "sources": list(set(all_sources)),
            "agent_responses": responses,
        }
