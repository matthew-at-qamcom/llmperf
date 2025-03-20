from datetime import datetime
from dataclasses import dataclass
import threading
import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import ray
import tqdm

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    randomly_sample_sonnet_lines_prompt,
    LLMPerfResults,
    sample_random_positive_int,
)
from tqdm import tqdm

from transformers import LlamaTokenizerFast


logger = logging.getLogger(__name__)


@dataclass
class PromptDetail:
    prompt: str
    num_input_tokens: int
    expected_num_output_tokens: int

def create_expected_num_input_tokens(means, normalised_stddev):
    results = []
    for mean in means:
        value = sample_random_positive_int(mean, mean*normalised_stddev)
        results.append(value)
    return results


def create_expected_num_output_tokens(size, mean, stddev):
    results = []
    for i in range(size):
        value = sample_random_positive_int(mean, stddev)
        results.append(value)
    return results


def create_prompts(expected_num_input_tokens_list, expected_num_output_tokens_list):
    start_time = time.time()
    logger.debug("Creating prompts")
    prompt_details: List[PromptDetail] = []
    zipped_list = list(zip(expected_num_input_tokens_list, expected_num_output_tokens_list))
    for num_input_tokens, num_output_tokens in tqdm(zipped_list, desc="Creating prompts"):
        prompt, num_prompt_tokens = randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean=num_input_tokens,
            prompt_tokens_stddev=0,
            expect_output_tokens=num_output_tokens,
            #tokenizer=tokenizer
        )
        prompt_detail = PromptDetail(
            prompt=prompt,
            num_input_tokens=num_input_tokens,
            expected_num_output_tokens=num_output_tokens,
        )
        prompt_details.append(prompt_detail)
        processing_time = time.time() - start_time
    logger.debug(f"Generated {len(prompt_details)} prompts in {processing_time:.1f} seconds")
    return prompt_details


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens_list: List[int],
    norm_stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: float,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    #random.seed(11111)

    # tokenizer = LlamaTokenizerFast.from_pretrained(
    #     "hf-internal-testing/llama-tokenizer",
    #     legacy=False
    # )
    # get_token_length = lambda text: len(tokenizer.encode(text))

    if not additional_sampling_params:
        additional_sampling_params = {}

    completed_requests_lock = threading.Lock()
    completed_requests = []
    num_completed_requests = 0
    # make up prompts outside of send loop for faster benchmarking loop
    expected_num_input_tokens = create_expected_num_input_tokens(
        means=mean_input_tokens_list*max_num_completed_requests,
        normalised_stddev=norm_stddev_input_tokens,
    )
    expected_num_output_tokens = create_expected_num_output_tokens(
        size=len(expected_num_input_tokens),
        mean=mean_output_tokens,
        stddev=stddev_output_tokens,
    )
    prompt_details = create_prompts(expected_num_input_tokens, expected_num_output_tokens)

    logger.debug("Processing requests")
    start_time = time.monotonic()
    num_requests = len(prompt_details)
    pbar = tqdm(total=num_requests, desc="Processing requests")

    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        request_index = thread_index % num_requests

        while (
            time.monotonic() - start_time < test_timeout_s
            and num_completed_requests < num_requests
        ):
            default_sampling_params = {"max_tokens": prompt_details[request_index].expected_num_output_tokens }
            default_sampling_params.update(additional_sampling_params)
            request_config = RequestConfig(
                model=model,
                prompt=(prompt_details[request_index].prompt,prompt_details[request_index].num_input_tokens),
                sampling_params=default_sampling_params,
                llm_api=llm_api,
            )
            req_launcher.launch_requests(request_config)

            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                with completed_requests_lock:
                    if num_completed_requests < num_requests:  # FIXME: Why is this condition necessary?
                        all_metrics.append(request_metrics)
                        completed_requests.extend(all_metrics)
                        pbar.update(len(all_metrics))
                        num_completed_requests += len(all_metrics)
                        request_index = (request_index + num_concurrent_requests) % num_requests

    threads = []
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pbar.close()

    # Check one last time that there are no remaining results to collect.
    clients = construct_clients(llm_api=llm_api, num_clients=1)
    req_launcher = RequestsLauncher(clients)
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, request_config = out
        with completed_requests_lock:
            if num_completed_requests < num_requests:
                completed_requests.extend(request_metrics)

    end_time = time.monotonic()
    processing_time = end_time - start_time
    logger.debug(f"Processed {len(completed_requests)} requests in {processing_time:.1f} seconds")
    if end_time - start_time >= test_timeout_s:
        logger.warning("Test timed out before all requests could be completed.")


    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        "mean_input_tokens_list": mean_input_tokens_list,
        "norm_stddev_input_tokens": norm_stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret

    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in [
        common_metrics.TTFT,
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_INPUT_THROUGHPUT_TOKENS,
        common_metrics.REQ_OUTPUT_THROUGHPUT_TOKENS,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        if len(series) == 1:
            print(f"    {series[0]}")
        else:
            quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
            quantiles_reformatted_keys = {}
            for quantile, value in quantiles.items():
                reformatted_key = f"p{int(quantile * 100)}"
                print(f"    {reformatted_key} = {value}")
                quantiles_reformatted_keys[reformatted_key] = value
            ret[key]["quantiles"] = quantiles_reformatted_keys
            mean = series.mean()
            print(f"    mean = {mean}")
            ret[key]["mean"] = mean
            print(f"    min = {series.min()}")
            ret[key]["min"] = series.min()
            print(f"    max = {series.max()}")
            ret[key]["max"] = series.max()
            print(f"    stddev = {series.std()}")
            ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    # overall_output_throughput = df_without_errored_req[
    #     common_metrics.NUM_OUTPUT_TOKENS
    # ].sum() / (end_time - start_time)

    # print(f"Overall Output Throughput: {overall_output_throughput}")
    # ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens_list: List[int],
    norm_stddev_input_tokens: float,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """
    if np.any(np.array(mean_input_tokens_list) < 40):
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        mean_input_tokens_list=mean_input_tokens_list,
        norm_stddev_input_tokens=norm_stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
    )

    if results_dir:
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        filename = f"{model}_{timestamp}"
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)

        # Update to metadata.
        summary.update(user_metadata)

        perf_results = LLMPerfResults(name=model, metadata=summary)
        results_dir = Path(results_dir)

        output = {
            "individual_responses": individual_responses,
            "summary": perf_results.to_dict()
        }

        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            filepath = results_dir / f"{filename}.json"
            print(f"Writing summary to {filepath}")
            with open(filepath, "w") as f:
                json.dump(output, f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e


args = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
args.add_argument(
    "--mean-input-tokens-list",
    type=int,
    nargs="+",
    default=[550],
    help=(
        "A list of the mean number of tokens to send in the prompt for the request. "
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--norm-stddev-input-tokens",
    type=float,
    default=0.1,
    help=(
        "The normalised standard deviation (where the mean is assumed to be one) of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--mean-output-tokens",
    type=int,
    default=150,
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--num-concurrent-requests",
    type=int,
    default=10,
    help=("The number of concurrent requests to send (default: %(default)s)"),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("filelock").setLevel(logging.INFO)

    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    args = args.parse_args()

    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    run_token_benchmark(
        llm_api=args.llm_api,
        model=args.model,
        test_timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        mean_input_tokens_list=args.mean_input_tokens_list,
        norm_stddev_input_tokens=args.norm_stddev_input_tokens,
        mean_output_tokens=args.mean_output_tokens,
        stddev_output_tokens=args.stddev_output_tokens,
        num_concurrent_requests=args.num_concurrent_requests,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
    )
