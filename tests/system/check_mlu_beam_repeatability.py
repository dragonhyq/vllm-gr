# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Check cold and cached OneRec requests against a running width-128 MLU GR server."""

import argparse
import json
import urllib.request
import uuid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/") + "/v1/completions"

    for history_size in (3, 22, 45):
        history = " ".join(
            f"<|sid_begin|><s_a_{i}><s_b_{i + 2}><s_c_{i + 4}><|sid_end|>"
            for i in range(history_size)
        )
        request = urllib.request.Request(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "model": args.model,
                    "prompt": f"Watch history: {history}. Recommend the next video.",
                    "cache_salt": uuid.uuid4().hex,
                    "use_beam_search": True,
                    "n": 128,
                    "temperature": 0.0,
                    "max_tokens": 5,
                    "ignore_eos": True,
                    "return_token_ids": True,
                    "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
                }
            ).encode(),
        )
        cold = None
        for run in range(3):
            with urllib.request.urlopen(request, timeout=120) as response:
                choices = json.load(response)["choices"]
            if len(choices) != 128 or any(len(choice["token_ids"]) != 5 for choice in choices):
                raise RuntimeError("Expected 128 complete five-token recommendations")
            ranked = [(choice["text"], choice["token_ids"]) for choice in choices]
            if cold is None:
                cold = ranked
            elif ranked != cold:
                rank = next(i for i, pair in enumerate(zip(cold, ranked), 1) if pair[0] != pair[1])
                raise RuntimeError(
                    f"History length {history_size}, run {run + 1} first differed at rank {rank}"
                )
        print(f"History length {history_size}: all 128 recommendations repeated exactly")


if __name__ == "__main__":
    main()
