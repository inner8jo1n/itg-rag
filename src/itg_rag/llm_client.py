import logging

from openai import OpenAI, omit

from itg_rag.api import call_api
from itg_rag.errors import RAGError

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def complete(
        self, system: str, user: str, *, json_mode: bool = True
    ) -> str:
        logger.info("Generating with %s; json_mode=%s", self.model, json_mode)
        response = call_api(
            lambda: self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=2200,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"} if json_mode else omit,
            )
        )
        if not response.choices:
            raise RAGError("LLM returned no choices")
        if response.choices[0].finish_reason != "stop":
            raise RAGError("LLM response was incomplete or refused")
        return response.choices[0].message.content or ""
