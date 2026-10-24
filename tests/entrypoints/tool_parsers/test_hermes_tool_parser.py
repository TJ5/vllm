# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai
import pytest
import pytest_asyncio
from typing_extensions import TypedDict

from tests.utils import RemoteOpenAIServer
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.granite4_tool_parser import Granite4ToolParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.transformers_utils.repo_utils import hf_api

LORA_MODEL = "minpeter/LoRA-Llama-3.2-1B-tool-vllm-ci"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]


class ServerConfig(TypedDict, total=False):
    model: str
    arguments: list[str]
    model_arg: str
    tool_parser: type[ToolParser]


CONFIGS: dict[str, ServerConfig] = {
    "llama": {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "arguments": [
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--enable-lora",
            "--lora-modules",
            f"{LORA_MODEL}={LORA_MODEL}",
            "--tokenizer",
            f"{LORA_MODEL}",
        ],
        "model_arg": LORA_MODEL,
        "tool_parser": Hermes2ProToolParser,
    },
    "granite4": {
        "model": "ibm-granite/granite-4.0-h-tiny",
        "arguments": [
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "granite4",
            "--tokenizer",
            "ibm-granite/granite-4.0-h-tiny",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "2",
        ],
        "model_arg": "ibm-granite/granite-4.0-h-tiny",
        "tool_parser": Granite4ToolParser,
    },
}


# for each server config, download the model and return the config
@pytest.fixture(scope="session", params=CONFIGS.keys())
def server_config(request):
    config = CONFIGS[request.param]

    # download model and tokenizer using transformers
    hf_api().snapshot_download(config["model"])
    yield CONFIGS[request.param]


@pytest.fixture(scope="module")
def server(request, server_config: ServerConfig):
    model = server_config["model"]
    args_for_model = server_config["arguments"]
    with RemoteOpenAIServer(model, args_for_model, max_wait_seconds=480) as server:
        yield server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    async with server.get_async_client() as async_client:
        yield async_client


PRODUCT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Get detailed information of a product based on its "
            "product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "inserted": {
                        "type": "boolean",
                        "description": "inserted.",
                    },
                    "product_id": {
                        "type": "integer",
                        "description": "The product ID of the product.",
                    },
                },
                "required": ["product_id", "inserted"],
            },
        },
    }
]

MESSAGES = [{"role": "user", "content": "What's the weather like in Boston?"}]

PRODUCT_MESSAGES = [
    {
        "role": "user",
        "content": "Hi! Do you have any detailed information about the product id "
        "7355608 and inserted true?",
    }
]


@pytest.mark.asyncio
async def test_non_streaming_tool_call(
    client: openai.AsyncOpenAI, server_config: ServerConfig
):
    """Test tool call in non-streaming mode."""
    response = await client.chat.completions.create(
        model=server_config["model_arg"],
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.0,
    )

    assert response.choices
    choice = response.choices[0]
    message = choice.message

    assert choice.finish_reason == "tool_calls"
    assert message.tool_calls is not None

    tool_call = message.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_current_weather"

    arguments = json.loads(tool_call.function.arguments)
    assert "location" in arguments
    assert "Boston" in arguments["location"]
    print("\n[Non-Streaming Test Passed]")
    print(f"Tool Call: {tool_call.function.name}")
    print(f"Arguments: {arguments}")


@pytest.mark.asyncio
async def test_streaming_tool_call(
    client: openai.AsyncOpenAI, server_config: ServerConfig
):
    """Test tool call in streaming mode."""
    stream = await client.chat.completions.create(
        model=server_config["model_arg"],
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.0,
        stream=True,
    )

    tool_call_chunks = {}
    async for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if not delta or not delta.tool_calls:
            continue

        for tool_chunk in delta.tool_calls:
            index = tool_chunk.index
            if index not in tool_call_chunks:
                tool_call_chunks[index] = {"name": "", "arguments": ""}

            if tool_chunk.function.name:
                tool_call_chunks[index]["name"] += tool_chunk.function.name
            if tool_chunk.function.arguments:
                tool_call_chunks[index]["arguments"] += tool_chunk.function.arguments

    assert len(tool_call_chunks) == 1
    reconstructed_tool_call = tool_call_chunks[0]

    assert reconstructed_tool_call["name"] == "get_current_weather"

    arguments = json.loads(reconstructed_tool_call["arguments"])
    assert "location" in arguments
    assert "Boston" in arguments["location"]
    print("\n[Streaming Test Passed]")
    print(f"Reconstructed Tool Call: {reconstructed_tool_call['name']}")
    print(f"Reconstructed Arguments: {arguments}")


@pytest.mark.asyncio
async def test_non_streaming_product_tool_call(
    client: openai.AsyncOpenAI, server_config: ServerConfig
):
    """Test tool call integer and boolean parameters in non-streaming mode."""
    response = await client.chat.completions.create(
        model=server_config["model_arg"],
        messages=PRODUCT_MESSAGES,
        tools=PRODUCT_TOOLS,
        tool_choice="auto",
        temperature=0.66,
    )

    assert response.choices
    choice = response.choices[0]
    message = choice.message

    assert choice.finish_reason == "tool_calls"
    assert message.tool_calls is not None

    tool_call = message.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_product_info"

    arguments = json.loads(tool_call.function.arguments)
    assert "product_id" in arguments
    assert "inserted" in arguments

    product_id = arguments.get("product_id")
    inserted = arguments.get("inserted")

    assert isinstance(product_id, int)
    assert product_id == 7355608
    assert isinstance(inserted, bool)
    assert inserted is True

    print("\n[Non-Streaming Product Test Passed]")
    print(f"Tool Call: {tool_call.function.name}")
    print(f"Arguments: {arguments}")


@pytest.mark.asyncio
async def test_streaming_product_tool_call(
    client: openai.AsyncOpenAI, server_config: ServerConfig
):
    """Test tool call integer and boolean parameters in streaming mode."""
    stream = await client.chat.completions.create(
        model=server_config["model_arg"],
        messages=PRODUCT_MESSAGES,
        tools=PRODUCT_TOOLS,
        tool_choice="auto",
        temperature=0.66,
        stream=True,
    )

    tool_call_chunks = {}
    async for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if not delta or not delta.tool_calls:
            continue

        for tool_chunk in delta.tool_calls:
            index = tool_chunk.index
            if index not in tool_call_chunks:
                tool_call_chunks[index] = {"name": "", "arguments": ""}

            if tool_chunk.function.name:
                tool_call_chunks[index]["name"] += tool_chunk.function.name
            if tool_chunk.function.arguments:
                tool_call_chunks[index]["arguments"] += tool_chunk.function.arguments

    assert len(tool_call_chunks) == 1
    reconstructed_tool_call = tool_call_chunks[0]

    assert reconstructed_tool_call["name"] == "get_product_info"

    arguments = json.loads(reconstructed_tool_call["arguments"])
    assert "product_id" in arguments
    assert "inserted" in arguments

    # Handle type coercion for streaming test as well
    product_id = arguments.get("product_id")
    inserted = arguments.get("inserted")

    assert isinstance(product_id, int)
    assert product_id == 7355608
    assert isinstance(inserted, bool)
    assert inserted is True

    print("\n[Streaming Product Test Passed]")
    print(f"Reconstructed Tool Call: {reconstructed_tool_call['name']}")
    print(f"Reconstructed Arguments: {arguments}")


@pytest.fixture(scope="module")
def qwen_json_server():
    args = [
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_json",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.1",
    ]
    with RemoteOpenAIServer("Qwen/Qwen3-0.6B", args) as server:
        yield server


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("choice", "parallel"),
    [("auto", True), ("required", True), ("required", False), ("named", True)],
)
async def test_qwen_json_parallel_tool_calls(
    qwen_json_server, stream, choice, parallel
):
    """Native Qwen framing must allow both cities under required tool choice."""
    tool_choice = (
        {"type": "function", "function": {"name": "get_current_weather"}}
        if choice == "named"
        else choice
    )
    async with qwen_json_server.get_async_client() as client:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-0.6B",
            messages=[
                {
                    "role": "user",
                    "content": "Get the current weather for both Paris and Tokyo. "
                    "Call the weather tool once for each city. "
                    "Do not write any other text.",
                }
            ],
            tools=TOOLS,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel,
            temperature=0,
            max_tokens=256,
            stream=stream,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if stream:
            calls: dict[int, tuple[str, str]] = {}
            content = ""
            finish_reason = None
            async for chunk in response:
                if not chunk.choices:
                    continue
                item = chunk.choices[0]
                content += item.delta.content or ""
                finish_reason = item.finish_reason or finish_reason
                for call in item.delta.tool_calls or []:
                    name, arguments = calls.get(call.index, ("", ""))
                    if call.function:
                        name += call.function.name or ""
                        arguments += call.function.arguments or ""
                    calls[call.index] = (name, arguments)
            parsed_calls = list(calls.values())
        else:
            item = response.choices[0]
            content = item.message.content or ""
            finish_reason = item.finish_reason
            parsed_calls = [
                (call.function.name, call.function.arguments)
                for call in item.message.tool_calls or []
            ]

    assert finish_reason == ("stop" if choice == "named" else "tool_calls")
    assert content == ""
    expected_count = 2 if parallel and choice != "named" else 1
    assert len(parsed_calls) == expected_count
    locations = []
    for name, arguments in parsed_calls:
        assert name == "get_current_weather"
        location = json.loads(arguments)["location"]
        assert "Paris" in location or "Tokyo" in location
        locations.append(location)
    if expected_count == 2:
        assert any("Paris" in location for location in locations)
        assert any("Tokyo" in location for location in locations)
