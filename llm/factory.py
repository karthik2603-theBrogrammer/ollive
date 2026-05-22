import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from transformers import GenerationConfig

from config import FrontierConfig, OSSConfig
from llm.cleaned_hf_chat import CleanedChatHuggingFace

logger = logging.getLogger(__name__)


def _configure_local_generation(llm: CleanedChatHuggingFace, config: OSSConfig) -> None:
    """Store generation settings on the pipeline config to avoid duplicate kwargs warnings."""
    hf_pipeline = getattr(llm.llm, "pipeline", None)
    if hf_pipeline is None:
        return

    hf_pipeline.generation_config = GenerationConfig(
        max_new_tokens=config.max_tokens,
        temperature=config.temperature,
        do_sample=config.temperature > 0,
    )

    runtime_kwargs = {
        "return_full_text": False,
        "clean_up_tokenization_spaces": False,
        "truncation": False,
    }
    llm.llm.pipeline_kwargs = runtime_kwargs


def build_oss_llm(config: OSSConfig) -> BaseChatModel:
    backend = config.backend.lower()

    if backend == "api":
        return _build_oss_api_llm(config)
    if backend == "local":
        return _build_oss_local_llm(config)

    raise ValueError(f"Unknown OSS_BACKEND '{config.backend}'. Use 'local' or 'api'.")


def _build_oss_local_llm(config: OSSConfig) -> BaseChatModel:
    """Run the model locally via transformers (works for small models like 0.5B)."""
    model_kwargs: dict = {}
    if config.hf_token:
        model_kwargs["token"] = config.hf_token

    pipeline_kwargs = {
        "return_full_text": False,
        "clean_up_tokenization_spaces": False,
        "truncation": False,
    }

    llm = CleanedChatHuggingFace.from_model_id(
        model_id=config.model_id,
        backend="pipeline",
        task="text-generation",
        device=config.device,
        model_kwargs=model_kwargs or None,
        pipeline_kwargs=pipeline_kwargs,
    )
    _configure_local_generation(llm, config)
    device_label = "CPU" if config.device < 0 else f"cuda:{config.device}"
    logger.info(
        "Built local OSS LangChain LLM model=%s device=%s",
        config.model_id,
        device_label,
    )
    return llm


def _build_oss_api_llm(config: OSSConfig) -> BaseChatModel:
    """Use Hugging Face Inference API (only models enabled on your HF account)."""
    if not config.hf_token:
        raise ValueError(
            "HF_TOKEN is required for OSS_BACKEND=api. "
            "Get one at https://huggingface.co/settings/tokens"
        )

    endpoint = HuggingFaceEndpoint(
        repo_id=config.model_id,
        huggingfacehub_api_token=config.hf_token,
        max_new_tokens=config.max_tokens,
        temperature=config.temperature,
        return_full_text=False,
        task="text-generation",
    )
    llm = CleanedChatHuggingFace(llm=endpoint)
    logger.info("Built HF API OSS LangChain LLM model=%s", config.model_id)
    return llm


def build_frontier_llm(config: FrontierConfig) -> BaseChatModel:
    provider = config.provider.lower()

    if provider == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when FRONTIER_PROVIDER=anthropic"
            )
        llm = ChatAnthropic(
            model=config.model_id,
            api_key=config.anthropic_api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    else:
        if not config.api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in ollive/.env")
        llm = ChatOpenAI(
            model=config.model_id,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    logger.info("Built frontier LangChain LLM provider=%s model=%s", provider, config.model_id)
    return llm
