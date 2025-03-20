import json
import os
import time
from typing import Any, Dict
import logging

import ray
import requests
import tiktoken

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


logger = logging.getLogger(__name__)

@ray.remote
class OpenAIChatCompletionsClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        times_to_next_token = []
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        output_start_time = None
        most_recent_received_token_time = None
        finish_reason = None
        num_input_tokens = None
        num_output_tokens = None
        num_total_tokens = None
        address = os.environ.get("OPENAI_API_BASE")
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("the environment variable OPENAI_API_KEY must be set.")
        headers = {"Authorization": f"Bearer {key}"}
        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"
        start_time = time.monotonic()
        try:
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=180,
                headers=headers,
            ) as response:
                #import curlify
                #curl_command = curlify.to_curl(response.request)
                #print(curl_command)

                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                for chunk in response.iter_lines(chunk_size=None):
                    if output_start_time is None:
                        output_start_time = time.monotonic()

                    chunk = chunk.strip()

                    if not chunk:
                        continue
                    stem = "data: "
                    chunk = chunk[len(stem) :]
                    if chunk == b"[DONE]":
                        continue
                    data = json.loads(chunk)

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        raise RuntimeError(data["error"]["message"])

                    if len(data["choices"]) >= 1:
                        choice_0 = data["choices"][0]
                        delta = choice_0["delta"]
                        if delta.get("content", None):
                            this_token_received_time = time.monotonic()
                            if most_recent_received_token_time is not None:
                                times_to_next_token.append(
                                    this_token_received_time - most_recent_received_token_time
                                )

                            most_recent_received_token_time = this_token_received_time
                            generated_text += delta["content"]

                        chunk_finish_reason = choice_0.get("finish_reason")
                        if chunk_finish_reason is not None:
                            finish_reason = chunk_finish_reason
                            final_chunk = chunk

                    usage = data.get("usage")
                    if usage is not None:
                        num_input_tokens = usage["prompt_tokens"]
                        num_output_tokens = usage["completion_tokens"]
                        num_total_tokens = usage["total_tokens"]

            logger.debug("Finish reason: %s", finish_reason)
            end_time = time.monotonic()
            ttft = output_start_time - start_time
            total_request_time = end_time - start_time
            output_duration = end_time - output_start_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        num_input_chars = len(prompt)
        num_output_chars = len(generated_text)
        metrics[common_metrics.NUM_INPUT_CHARS] = num_input_chars
        metrics[common_metrics.NUM_OUTPUT_CHARS] = num_output_chars
        metrics[common_metrics.NUM_TOTAL_CHARS] = num_input_chars + num_output_chars
        metrics[common_metrics.NUM_INPUT_TOKENS] = num_input_tokens
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
        metrics[common_metrics.NUM_TOTAL_TOKENS] = num_total_tokens
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.OUTPUT_DURATION] = output_duration
        metrics[common_metrics.E2E_LAT] = total_request_time

        if num_output_tokens:
            metrics[common_metrics.INTER_TOKEN_LAT] = (
                output_duration / num_output_tokens
            )
        else:
            metrics[common_metrics.INTER_TOKEN_LAT] = None
        metrics[common_metrics.REQ_INPUT_THROUGHPUT_CHARS] = (
            num_input_chars / ttft
        )
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT_CHARS] = (
            num_output_chars / output_duration
        )
        metrics[common_metrics.REQ_INPUT_THROUGHPUT_TOKENS] = (
            num_input_tokens / ttft
        )
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT_TOKENS] = (
            num_output_tokens / output_duration
        )

        return metrics, generated_text, request_config
