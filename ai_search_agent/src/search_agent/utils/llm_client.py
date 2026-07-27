"""
Thin wrapper around ChatGroq (via langchain_groq) for forced structured output.

Why `.with_structured_output()` instead of raw prompting:
    LangChain's `.with_structured_output(schema)` binds the Pydantic model
    directly to the chat model, returning a validated Python object instead of
    raw text that we'd have to parse.  On validation failure we retry with the
    error fed back to the model so it can self-correct.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from search_agent.config import LLMSettings
from search_agent.config import settings as default_settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredExtractionError(RuntimeError):
    """Raised when the LLM's structured output could not be parsed/validated
    after exhausting all configured retry attempts."""


@dataclass(frozen=True)
class StructuredCompletion:
    """Validated structured-output result plus lightweight observability
    metadata, useful for logging/tracing as this gets wired into LangGraph."""

    parsed: BaseModel
    raw_tool_input: Dict[str, Any]
    model: str
    attempt: int


class GroqStructuredClient:
    """Forces schema-conformant structured output from a Groq-hosted model and
    validates it against a caller-supplied Pydantic model, retrying with the
    validation error fed back to the model on failure.
    """

    def __init__(
        self,
        llm_settings: Optional[LLMSettings] = None,
        client: Optional[ChatGroq] = None,
    ) -> None:
        self.settings = llm_settings or default_settings.llm
        # Allow an injected client (handy for tests); otherwise build from settings.
        self._client: ChatGroq = client or ChatGroq(
            model=self.settings.model,
            temperature=self.settings.temperature,
            api_key=os.getenv(self.settings.api_key_env_var) or None,  # type: ignore[arg-type]
        )

    def extract_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[SchemaT],
        tool_name: str = "extract_structured_output",
        tool_description: str = "Return the structured extraction result.",
    ) -> StructuredCompletion:
        """Calls the model, forcing it to return output conforming to
        `schema_model`, then validates the result.

        Retries up to `settings.max_retries` additional times on validation
        failure (total attempts = max_retries + 1), feeding the validation
        error back into the conversation so the model can self-correct.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user turn content.
            schema_model: Pydantic model class that defines the expected output shape.
            tool_name: Unused by Groq path but kept for API compatibility.
            tool_description: Unused by Groq path but kept for API compatibility.

        Returns:
            A :class:`StructuredCompletion` with the validated parsed object.

        Raises:
            StructuredExtractionError: If all retry attempts are exhausted.
        """
        structured_llm = self._client.with_structured_output(schema_model)

        messages: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        last_error: Optional[Exception] = None
        total_attempts = self.settings.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            logger.debug(
                "Calling model=%s attempt=%d/%d",
                self.settings.model, attempt, total_attempts,
            )
            try:
                result = structured_llm.invoke(messages)

                # `with_structured_output` already validates against the schema,
                # but we call model_validate() as a belt-and-suspenders check.
                if not isinstance(result, schema_model):
                    result = schema_model.model_validate(
                        result.model_dump() if hasattr(result, "model_dump") else result
                    )

                raw = result.model_dump()
                return StructuredCompletion(
                    parsed=result,
                    raw_tool_input=raw,
                    model=self.settings.model,
                    attempt=attempt,
                )
            except (ValidationError, Exception) as exc:
                last_error = exc
                logger.warning("Attempt %d/%d: failed: %s", attempt, total_attempts, exc)
                if attempt < total_attempts:
                    # Feed the error back so the model can self-correct on retry.
                    messages.append(
                        HumanMessage(
                            content=(
                                f"Your previous response failed validation with this error:\n{exc}\n"
                                "Please try again with a corrected response."
                            )
                        )
                    )

        raise StructuredExtractionError(
            f"Failed to get valid structured output after {total_attempts} attempt(s). "
            f"Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Backwards-compatible alias so existing callers (stage1_intent_extraction,
# tests) don't need immediate changes.
# ---------------------------------------------------------------------------
AnthropicStructuredClient = GroqStructuredClient
