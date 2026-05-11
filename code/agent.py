import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are a professional climate change evidence assistant.\n\n"
    "Your task is to provide clear, concise, and scientifically accurate evidence of climate change, using only the information retrieved from the authorized knowledge base document (Climate.pdf).\n\n"
    "- Begin with a brief summary of what climate change is.\n\n"
    "- Present specific, referenced evidence of climate change, such as observed temperature increases, melting ice, rising sea levels, and increased extreme weather events.\n\n"
    "- Attribute all evidence to the knowledge base content.\n\n"
    "- If the knowledge base does not contain relevant evidence, respond with a polite message indicating that no evidence was found.\n\n"
    "- Maintain a formal and objective tone.\n\n"
    "Output format:\n\n"
    "- Summary paragraph\n"
    "- Bullet points listing key evidence\n"
    "- Reference to the source document (Climate.pdf)\n\n"
    "Fallback: \"If no evidence is found in the knowledge base, reply: \\\"No evidence of\" climate change was found in the available knowledge base content.\""
)
OUTPUT_FORMAT = "- Summary paragraph\n- Bullet points of evidence\n- Reference to source document"
FALLBACK_RESPONSE = "No evidence of climate change was found in the available knowledge base content."
SELECTED_DOCUMENT_TITLES = ["Climate.pdf"]
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# LOGGING CONFIGURATION
# =========================

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# =========================
# SANITIZER UTILITY
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")


def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()


@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# FASTAPI OBSERVABILITY LIFESPAN
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Climate Change Evidence Assistant",
    description="Provides evidence-based answers about climate change using only authorized knowledge base content (Climate.pdf).",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# =========================
# DATA MODELS
# =========================

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully")
    answer: Optional[str] = Field(None, description="Evidence-based answer or fallback message")
    error: Optional[str] = Field(None, description="Error message, if any")
    error_code: Optional[str] = Field(None, description="Error code, if any")

# =========================
# AZURE AI SEARCH CLIENT
# =========================

class AzureAISearchClient:
    """Handles low-level API calls to Azure AI Search."""

    def __init__(self):
        self._client = None

    def get_client(self) -> SearchClient:
        """Lazily initialize and return the SearchClient."""
        if self._client is not None:
            return self._client
        endpoint = Config.AZURE_SEARCH_ENDPOINT
        api_key = Config.AZURE_SEARCH_API_KEY
        index_name = Config.AZURE_SEARCH_INDEX_NAME
        if not endpoint or not api_key or not index_name:
            raise RuntimeError("Azure AI Search configuration missing (endpoint, api_key, or index_name)")
        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key),
        )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def search(self, query: str, filter: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        """Perform vector + keyword search with optional OData filter."""
        search_client = self.get_client()
        # Get embedding for the query
        openai_client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        _t0 = _time.time()
        embedding_resp = await openai_client.embeddings.create(
            input=query,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        )
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary="embedding",
            )
        except Exception:
            pass

        vector_query = VectorizedQuery(
            vector=embedding_resp.data[0].embedding,
            k_nearest_neighbors=top_k,
            fields="vector"
        )

        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": ["chunk", "title"],
        }
        if filter:
            search_kwargs["filter"] = filter

        _t1 = _time.time()
        results = search_client.search(**search_kwargs)
        try:
            trace_tool_call(
                tool_name="search_client.search",
                latency_ms=int((_time.time() - _t1) * 1000),
                output=str(results)[:200] if results is not None else None,
                status="success",
            )
        except Exception:
            pass

        chunks = []
        for r in results:
            if r.get("chunk"):
                chunks.append({"chunk": r["chunk"], "title": r.get("title")})
        return chunks

# =========================
# CHUNK RETRIEVER
# =========================

class ChunkRetriever:
    """Queries Azure AI Search for relevant chunks from Climate.pdf."""

    def __init__(self, azure_search_client: AzureAISearchClient):
        self.azure_search_client = azure_search_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, filter: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks, filtered to Climate.pdf."""
        async with trace_step(
            "retrieve_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant chunks from Azure AI Search filtered to Climate.pdf",
            output_fn=lambda r: f"{len(r)} chunks"
        ) as step:
            try:
                chunks = await self.azure_search_client.search(query, filter, top_k)
                # Validate all chunks are from Climate.pdf
                for c in chunks:
                    if c.get("title") != "Climate.pdf":
                        logger.error(f"Retrieved chunk from unauthorized document: {c.get('title')}")
                        raise RuntimeError("DOCUMENT_RETRIEVAL_ERROR: Unauthorized document retrieved")
                step.capture(chunks)
                return chunks
            except Exception as e:
                logger.error(f"Error retrieving chunks: {e}")
                raise RuntimeError("DOCUMENT_RETRIEVAL_ERROR") from e

# =========================
# LLM SERVICE
# =========================

class LLMService:
    """Calls Azure OpenAI with system prompt, user query, and context."""

    def __init__(self):
        self._client = None

    def get_client(self):
        if self._client is not None:
            return self._client
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY not configured")
        self._client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def generate_response(self, system_prompt: str, user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate evidence-based response using LLM."""
        async with trace_step(
            "generate_response",
            step_type="llm_call",
            decision_summary="Call LLM with system prompt and retrieved chunks",
            output_fn=lambda r: f"LLM response: {str(r)[:80]}"
        ) as step:
            if not context_chunks:
                logger.info("No context chunks found; returning fallback response.")
                step.capture(FALLBACK_RESPONSE)
                return FALLBACK_RESPONSE

            context_text = "\n\n".join(c["chunk"] for c in context_chunks if c.get("chunk"))
            messages = [
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\nOutput Format: {OUTPUT_FORMAT}"
                },
                {
                    "role": "user",
                    "content": f"{user_query}\n\nContext:\n{context_text}"
                }
            ]
            _t0 = _time.time()
            client = self.get_client()
            _llm_kwargs = Config.get_llm_kwargs()
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4o",
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4o",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            sanitized = sanitize_llm_output(content, content_type="text")
            step.capture(sanitized)
            return sanitized

# =========================
# ERROR HANDLER
# =========================

class ErrorHandler:
    """Handles error detection, retry logic, fallback responses, and error code mapping."""

    def __init__(self, logger):
        self.logger = logger

    async def handle_error(self, error: Exception, context: dict = None) -> Dict[str, Any]:
        """Maps exceptions to error codes and returns fallback or error messages."""
        context = context or {}
        error_code = None
        message = None
        if isinstance(error, RuntimeError):
            msg = str(error)
            if "DOCUMENT_RETRIEVAL_ERROR" in msg:
                error_code = "DOCUMENT_RETRIEVAL_ERROR"
                message = "An error occurred while retrieving evidence from the knowledge base. Please try again later."
            elif "NO_EVIDENCE_FOUND" in msg:
                error_code = "NO_EVIDENCE_FOUND"
                message = FALLBACK_RESPONSE
            else:
                error_code = "AGENT_ERROR"
                message = "An unexpected error occurred. Please try again later."
        else:
            error_code = "AGENT_ERROR"
            message = "An unexpected error occurred. Please try again later."

        self.logger.error(f"Error handled: {error_code} - {message} | Details: {str(error)} | Context: {context}")
        return {
            "success": False,
            "answer": None,
            "error": message,
            "error_code": error_code
        }

# =========================
# LOGGER UTILITY
# =========================

class AgentLogger:
    """Logs all requests, responses, errors, and audit events."""

    def __init__(self):
        self.logger = logging.getLogger("agent")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: dict):
        try:
            self.logger.info(f"{event_type}: {json.dumps(details, default=str)}")
        except Exception as e:
            self.logger.warning(f"Logging error: {e}")

# =========================
# MAIN AGENT CLASS
# =========================

class ClimateChangeEvidenceAgent:
    """Orchestrates the flow: retrieval, LLM call, error handling, and logging."""

    def __init__(self):
        self.logger = AgentLogger()
        self.error_handler = ErrorHandler(self.logger)
        self.azure_search_client = AzureAISearchClient()
        self.chunk_retriever = ChunkRetriever(self.azure_search_client)
        self.llm_service = LLMService()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_query(self) -> Dict[str, Any]:
        """Main entry point. Orchestrates retrieval and LLM call, applies business rules, returns formatted response."""
        async with trace_step(
            "process_query",
            step_type="process",
            decision_summary="Orchestrate retrieval and LLM call for climate change evidence",
            output_fn=lambda r: f"success={r.get('success', False)}"
        ) as step:
            try:
                # Build OData filter for Climate.pdf
                odata_parts = [f"title eq '{t}'" for t in SELECTED_DOCUMENT_TITLES]
                filter_str = " or ".join(odata_parts) if odata_parts else None

                # Retrieve relevant chunks
                chunks = await self.chunk_retriever.retrieve_chunks(
                    query=SYSTEM_PROMPT,
                    filter=filter_str,
                    top_k=5
                )
                self.logger.log_event("retrieval", {"num_chunks": len(chunks)})

                # If no chunks, return fallback
                if not chunks:
                    self.logger.log_event("fallback", {"reason": "No evidence found in knowledge base"})
                    step.capture({"success": True, "answer": FALLBACK_RESPONSE})
                    return {
                        "success": True,
                        "answer": FALLBACK_RESPONSE,
                        "error": None,
                        "error_code": None
                    }

                # Call LLM to generate response
                answer = await self.llm_service.generate_response(
                    system_prompt=SYSTEM_PROMPT,
                    user_query=SYSTEM_PROMPT,
                    context_chunks=chunks
                )
                sanitized_answer = sanitize_llm_output(answer, content_type="text")
                self.logger.log_event("llm_response", {"answer": sanitized_answer[:200]})
                step.capture({"success": True, "answer": sanitized_answer})
                return {
                    "success": True,
                    "answer": sanitized_answer,
                    "error": None,
                    "error_code": None
                }
            except Exception as e:
                # Centralized error handling
                error_result = await self.error_handler.handle_error(e)
                step.capture(error_result)
                return error_result

# =========================
# FASTAPI ENDPOINTS
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint():
    """
    Endpoint for retrieving evidence of climate change.
    No user input is required; the agent uses the enhanced system prompt and document filter internally.
    """
    agent = ClimateChangeEvidenceAgent()
    try:
        result = await agent.process_query()
        return QueryResponse(**result)
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "answer": None,
                "error": "Invalid request format.",
                "error_code": "VALIDATION_ERROR"
            }
        )
    except Exception as e:
        logger.error(f"Unhandled error in /query endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "answer": None,
                "error": "Internal server error.",
                "error_code": "INTERNAL_ERROR"
            }
        )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "answer": None,
            "error": "Malformed JSON or invalid request format. Please check your input.",
            "error_code": "VALIDATION_ERROR"
        }
    )

@app.exception_handler(json.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    logger.error(f"JSON decode error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "answer": None,
            "error": "Malformed JSON in request body. Please ensure your JSON is valid.",
            "error_code": "JSON_DECODE_ERROR"
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "answer": None,
            "error": "An unexpected error occurred.",
            "error_code": "INTERNAL_ERROR"
        }
    )

# =========================
# AGENT ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())