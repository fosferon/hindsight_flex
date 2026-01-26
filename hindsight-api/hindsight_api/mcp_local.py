"""
Local MCP server for use with Claude Code (stdio transport).

This runs a fully local Hindsight instance with embedded PostgreSQL (pg0).
No external database or server required.

Run with:
    hindsight-local-mcp

Or with uvx:
    uvx hindsight-api@latest hindsight-local-mcp

Configure in Claude Code's MCP settings:
    {
        "mcpServers": {
            "hindsight": {
                "command": "uvx",
                "args": ["hindsight-api@latest", "hindsight-local-mcp"],
                "env": {
                    "HINDSIGHT_API_LLM_API_KEY": "your-openai-key"
                }
            }
        }
    }

Environment variables:
    HINDSIGHT_API_LLM_API_KEY: Required. API key for LLM provider.
    HINDSIGHT_API_LLM_PROVIDER: Optional. LLM provider (default: "openai").
    HINDSIGHT_API_LLM_MODEL: Optional. LLM model (default: "gpt-4o-mini").
    HINDSIGHT_API_MCP_LOCAL_BANK_ID: Optional. Memory bank ID (default: "mcp").
    HINDSIGHT_API_LOG_LEVEL: Optional. Log level (default: "warning").
    HINDSIGHT_API_MCP_INSTRUCTIONS: Optional. Additional instructions appended to both retain and recall tools.
    HINDSIGHT_API_DATABASE_URL: Optional. External PostgreSQL URL (default: uses embedded pg0).
    HINDSIGHT_API_SCHEMA: Optional. PostgreSQL schema for table isolation (default: public).

Example custom instructions (these are ADDED to the default behavior):
    To also store assistant actions:
        HINDSIGHT_API_MCP_INSTRUCTIONS="Also store every action you take, including tool calls, code written, and decisions made."

    To also store conversation summaries:
        HINDSIGHT_API_MCP_INSTRUCTIONS="Also store summaries of important conversations and their outcomes."
"""

import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import Icon

from hindsight_api.config import (
    DEFAULT_MCP_LOCAL_BANK_ID,
    DEFAULT_MCP_RECALL_DESCRIPTION,
    DEFAULT_MCP_RETAIN_DESCRIPTION,
    ENV_MCP_INSTRUCTIONS,
    ENV_MCP_LOCAL_BANK_ID,
)

# Configure logging - default to warning to avoid polluting stderr during MCP init
# MCP clients interpret stderr output as errors, so we suppress INFO logs by default
_log_level_str = os.environ.get("HINDSIGHT_API_LOG_LEVEL", "warning").lower()
_log_level_map = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
logging.basicConfig(
    level=_log_level_map.get(_log_level_str, logging.WARNING),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stderr,  # MCP uses stdout for protocol, logs go to stderr
)
logger = logging.getLogger(__name__)


def create_local_mcp_server(bank_id: str, memory=None) -> FastMCP:
    """
    Create a stdio MCP server with retain/recall/reflect/list_banks/create_bank tools.

    Args:
        bank_id: The memory bank ID to use for all operations.
        memory: Optional MemoryEngine instance. If not provided, creates one with pg0.

    Returns:
        Configured FastMCP server instance.
    """
    # Import here to avoid slow startup if just checking --help
    from hindsight_api import MemoryEngine
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES
    from hindsight_api.models import RequestContext

    # Create memory engine with pg0 embedded database if not provided
    if memory is None:
        memory = MemoryEngine(db_url="pg0://hindsight-mcp")

    # Get custom instructions from environment variable (appended to both tools)
    extra_instructions = os.environ.get(ENV_MCP_INSTRUCTIONS, "")

    retain_description = DEFAULT_MCP_RETAIN_DESCRIPTION
    recall_description = DEFAULT_MCP_RECALL_DESCRIPTION

    if extra_instructions:
        retain_description = f"{DEFAULT_MCP_RETAIN_DESCRIPTION}\n\nAdditional instructions: {extra_instructions}"
        recall_description = f"{DEFAULT_MCP_RECALL_DESCRIPTION}\n\nAdditional instructions: {extra_instructions}"

    mcp = FastMCP("hindsight")

    @mcp.tool(description=retain_description)
    async def retain(content: str, context: str = "general", target_bank: str | None = None) -> dict:
        """
        Args:
            content: The fact/memory to store (be specific and include relevant details)
            context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
            target_bank: Optional bank ID to store in. If not provided, uses the configured default bank.
        """
        import asyncio

        effective_bank = target_bank or bank_id

        async def _retain():
            try:
                await memory.retain_batch_async(
                    bank_id=effective_bank,
                    contents=[{"content": content, "context": context}],
                    request_context=RequestContext(),
                )
            except Exception as e:
                logger.error(f"Error storing memory: {e}", exc_info=True)

        # Fire and forget - don't block on memory storage
        asyncio.create_task(_retain())
        return {"status": "accepted", "message": f"Memory storage initiated (bank: {effective_bank})"}

    @mcp.tool(description=recall_description)
    async def recall(query: str, max_tokens: int = 4096, budget: str = "low", target_bank: str | None = None) -> dict:
        """
        Args:
            query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
            max_tokens: Maximum tokens to return in results (default: 4096)
            budget: Search budget level - "low", "mid", or "high" (default: "low")
            target_bank: Optional bank ID to search in. If not provided, uses the configured default bank.
        """
        try:
            effective_bank = target_bank or bank_id

            # Map string budget to enum
            budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
            budget_enum = budget_map.get(budget.lower(), Budget.LOW)

            search_result = await memory.recall_async(
                bank_id=effective_bank,
                query=query,
                fact_type=list(VALID_RECALL_FACT_TYPES),
                budget=budget_enum,
                max_tokens=max_tokens,
                request_context=RequestContext(),
            )

            return search_result.model_dump()
        except Exception as e:
            logger.error(f"Error searching: {e}", exc_info=True)
            return {"error": str(e), "results": []}

    @mcp.tool()
    async def reflect(query: str, context: str | None = None, budget: str = "low", target_bank: str | None = None) -> dict:
        """
        Generate thoughtful analysis by synthesizing stored memories with the bank's personality.

        WHEN TO USE THIS TOOL:
        Use reflect when you need reasoned analysis, not just fact retrieval. This tool
        thinks through the question using everything the bank knows and its personality traits.

        EXAMPLES OF GOOD QUERIES:
        - "What patterns have emerged in how I approach debugging?"
        - "Based on my past decisions, what architectural style do I prefer?"
        - "What might be the best approach for this problem given what you know about me?"
        - "How should I prioritize these tasks based on my goals?"

        HOW IT DIFFERS FROM RECALL:
        - recall: Returns raw facts matching your search (fast lookup)
        - reflect: Reasons across memories to form a synthesized answer (deeper analysis)

        Use recall for "what did I say about X?" and reflect for "what should I do about X?"

        Args:
            query: The question or topic to reflect on
            context: Optional context about why this reflection is needed
            budget: Search budget - 'low', 'mid', or 'high' (default: 'low')
            target_bank: Optional bank ID to reflect in. If not provided, uses the configured default bank.
        """
        try:
            effective_bank = target_bank or bank_id

            # Map string budget to enum
            budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
            budget_enum = budget_map.get(budget.lower(), Budget.LOW)

            reflect_result = await memory.reflect_async(
                bank_id=effective_bank,
                query=query,
                budget=budget_enum,
                context=context,
                request_context=RequestContext(),
            )

            return reflect_result.model_dump()
        except Exception as e:
            logger.error(f"Error reflecting: {e}", exc_info=True)
            return {"error": str(e), "text": ""}

    @mcp.tool()
    async def list_banks() -> dict:
        """
        List all available memory banks.

        Use this tool to discover what memory banks exist in the system.
        Each bank is an isolated memory store (like a separate "brain").

        Returns:
            List of banks with their IDs, names, dispositions, and missions.
        """
        try:
            banks = await memory.list_banks(request_context=RequestContext())
            return {"banks": banks}
        except Exception as e:
            logger.error(f"Error listing banks: {e}", exc_info=True)
            return {"error": str(e), "banks": []}

    @mcp.tool()
    async def create_bank(bank_id: str, name: str | None = None, mission: str | None = None) -> dict:
        """
        Create a new memory bank or get an existing one.

        Memory banks are isolated stores - each one is like a separate "brain" for a user/agent.
        Banks are auto-created with default settings if they don't exist.

        Args:
            bank_id: Unique identifier for the bank (e.g., 'user-123', 'agent-alpha')
            name: Optional human-friendly name for the bank
            mission: Optional mission describing who the agent is and what they're trying to accomplish
        """
        try:
            # get_bank_profile auto-creates bank if it doesn't exist
            profile = await memory.get_bank_profile(bank_id, request_context=RequestContext())

            # Update name/mission if provided
            if name is not None or mission is not None:
                await memory.update_bank(
                    bank_id,
                    name=name,
                    mission=mission,
                    request_context=RequestContext(),
                )
                # Fetch updated profile
                profile = await memory.get_bank_profile(bank_id, request_context=RequestContext())

            # Serialize disposition if it's a Pydantic model
            if "disposition" in profile and hasattr(profile["disposition"], "model_dump"):
                profile["disposition"] = profile["disposition"].model_dump()
            return profile
        except Exception as e:
            logger.error(f"Error creating bank: {e}", exc_info=True)
            return {"error": str(e)}

    return mcp


async def _initialize_and_run(bank_id: str, db_url: str, schema: str | None):
    """Initialize memory and run the MCP server."""
    from hindsight_api import MemoryEngine

    # Create and initialize memory engine
    # Uses pg0 embedded database by default, or external PostgreSQL if configured
    # Note: We avoid printing to stderr during init as MCP clients show it as "errors"
    memory = MemoryEngine(db_url=db_url, schema=schema)
    await memory.initialize()

    # Create and run the server
    mcp = create_local_mcp_server(bank_id, memory=memory)
    await mcp.run_stdio_async()


def main():
    """Main entry point for the stdio MCP server."""
    import asyncio

    from hindsight_api.config import ENV_LLM_API_KEY, get_config

    # Check for required environment variables
    config = get_config()
    if not config.llm_api_key:
        print(f"Error: {ENV_LLM_API_KEY} environment variable is required", file=sys.stderr)
        print("Set it in your MCP configuration or shell environment", file=sys.stderr)
        sys.exit(1)

    # Get bank ID from environment, default to "mcp"
    bank_id = os.environ.get(ENV_MCP_LOCAL_BANK_ID, DEFAULT_MCP_LOCAL_BANK_ID)

    # Get database URL from config (defaults to pg0://hindsight-mcp if not set)
    db_url = config.database_url if config.database_url != "pg0" else "pg0://hindsight-mcp"

    # Get schema from config (None = use public schema)
    schema = config.schema

    # Note: We don't print to stderr as MCP clients display it as "error output"
    # Use HINDSIGHT_API_LOG_LEVEL=debug for verbose startup logging

    # Run the async initialization and server
    asyncio.run(_initialize_and_run(bank_id, db_url, schema))


if __name__ == "__main__":
    main()
