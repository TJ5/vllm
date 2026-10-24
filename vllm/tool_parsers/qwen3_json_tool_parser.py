# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.tool_parsers.utils import partial_tag_overlap


class Qwen3JsonToolParser(Hermes2ProToolParser):
    """Qwen2.5/QwQ/Qwen3 JSON calls with template newlines as framing."""

    structural_tag_model = "qwen_3"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._pending_content = ""

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        result = super().extract_tool_calls(model_output, request)
        if result.tools_called and result.content:
            result.content = result.content.removesuffix("\n") or None
        return result

    def _extract_content(self, current_text: str) -> str | None:
        start = current_text.find(self.tool_call_start_token)
        if start == -1:
            overlap = partial_tag_overlap(current_text, self.tool_call_start_token)
            end = len(current_text) - overlap
        else:
            end = start
        # Hold exactly one newline until we know whether a tool call follows.
        if end and current_text[end - 1] == "\n":
            end -= 1
        self._pending_content = current_text[end:] if start == -1 else ""
        content = current_text[self._sent_content_idx : end]
        self._sent_content_idx = end
        return content or None

    def finish_streaming(self) -> DeltaMessage | None:
        content = self._pending_content
        self._pending_content = ""
        return DeltaMessage(content=content) if content else None
