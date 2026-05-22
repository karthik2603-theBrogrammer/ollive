import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import ChatHuggingFace

from assistants.response_utils import clean_model_response

logger = logging.getLogger(__name__)

GENERATION_PIPELINE_KEYS = frozenset(
    {
        "max_new_tokens",
        "max_length",
        "min_length",
        "temperature",
        "top_p",
        "top_k",
        "do_sample",
        "repetition_penalty",
        "generation_config",
    }
)


class CleanedChatHuggingFace(ChatHuggingFace):
    """
    ChatHuggingFace wrapper for local HF pipelines.

    Uses return_full_text=False so the pipeline returns only new tokens.
    Do NOT combine that with skip_prompt=True — it would slice away the reply.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        pipeline_kwargs = dict(getattr(self.llm, "pipeline_kwargs", None) or {})
        pipeline_kwargs["return_full_text"] = False
        for key in GENERATION_PIPELINE_KEYS:
            pipeline_kwargs.pop(key, None)

        kwargs["pipeline_kwargs"] = {**pipeline_kwargs, **kwargs.get("pipeline_kwargs", {})}
        for key in GENERATION_PIPELINE_KEYS:
            kwargs["pipeline_kwargs"].pop(key, None)

        result = super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            stream=stream,
            **kwargs,
        )

        cleaned_generations: list[ChatGeneration] = []
        for generation in result.generations:
            if not isinstance(generation, ChatGeneration):
                cleaned_generations.append(generation)
                continue

            raw = ""
            if generation.message and generation.message.content:
                raw = str(generation.message.content)
            elif generation.text:
                raw = generation.text

            content = clean_model_response(raw)
            if not content and raw:
                logger.warning("Cleaner removed all text; using raw model output")
                content = raw.strip()

            cleaned_generations.append(
                ChatGeneration(
                    message=AIMessage(content=content),
                    generation_info=generation.generation_info,
                )
            )

        result.generations = cleaned_generations
        return result
