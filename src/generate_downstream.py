from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List

from common import CONFIG_DIR, PROMPT_DIR, load_json
from generate_samples import load_model_config

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

DOWNSTREAM_PROMPT_DIR = PROMPT_DIR / "downstream"
DOWNSTREAM_CONFIG = load_json(CONFIG_DIR / "downstream_tasks.json")

MODEL_ALIASES = {
    "DeepSeek": "DeepSeek-V3.2",
    "DeepSeek-V3.2": "DeepSeek-V3.2",
    "GPT-4o": "GPT-4o",
    "Qwen3": "Qwen3",
    "Llama-3.3": "Llama-3.3",
    "Llama-4": "Llama-4",
    "GPT-OSS": "GPT-OSS",
}


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def load_prompt_pair(task: str) -> tuple[str, str]:
    system = (DOWNSTREAM_PROMPT_DIR / f"{task}_system.txt").read_text(encoding="utf-8").strip()
    user = (DOWNSTREAM_PROMPT_DIR / f"{task}_user.txt").read_text(encoding="utf-8").strip()
    return system, user


def make_openai_client(model_cfg: Dict[str, str]):
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    api_key = os.getenv(model_cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing environment variable: {model_cfg['api_key_env']}")
    kwargs = {"api_key": api_key}
    if model_cfg.get("base_url"):
        kwargs["base_url"] = model_cfg["base_url"]
    return OpenAI(**kwargs)


def call_downstream(model_name: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    model_cfg = load_model_config(canonical_model_name(model_name))
    provider = model_cfg["provider"]
    if provider == "openai_compatible":
        client = make_openai_client(model_cfg)
        response = client.chat.completions.create(
            model=model_cfg["api_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    if provider == "gemini":
        if genai is None:
            raise RuntimeError("google-genai package is not installed")
        api_key = os.getenv(model_cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(f"Missing environment variable: {model_cfg['api_key_env']}")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_cfg["api_model"],
            contents=f"System: {system_prompt}\n\nUser: {user_prompt}",
            config={"temperature": temperature, "top_p": 1.0},
        )
        return getattr(response, "text", "") or ""
    raise RuntimeError(f"Unsupported provider: {provider}")


def parse_mcq(text: str) -> dict:
    match = re.search(r"Correct Answer[：:]\s*([A-D])", text, re.IGNORECASE)
    return {"content": text, "parsed_answer": match.group(1).upper() if match else None}


def parse_joint(text: str) -> dict | None:
    clean = text.strip()
    lower = clean.lower()
    simple = re.search(r"\[(male|female)\]\s*\[(white|hispanic|black|asian|others?)\]\s*\[(\d{2,3})\]\s*\[(black|white|red|blue|green|yellow|brown)\]", lower)
    if simple:
        race = simple.group(2)
        if race == "other":
            race = "others"
        return {
            "gender": simple.group(1),
            "race": race,
            "height": int(simple.group(3)),
            "color": simple.group(4),
            "raw_output": clean,
        }
    detailed = re.search(
        r"(male|female).*?(white|hispanic|black|asian|others?).*?(\d{2,3}).*?(black|white|red|blue|green|yellow|brown)",
        lower,
        re.DOTALL,
    )
    if detailed:
        race = detailed.group(2)
        if race == "other":
            race = "others"
        height = int(detailed.group(3))
        if 100 <= height <= 250:
            return {
                "gender": detailed.group(1),
                "race": race,
                "height": height,
                "color": detailed.group(4),
                "raw_output": clean,
            }
    return None


def parse_gender(text: str) -> dict | None:
    match = re.search(r"\[(male|female)\]", text, re.IGNORECASE)
    if not match:
        return None
    return {"gender": match.group(1).lower(), "prompt": text.strip()}


def parse_race(text: str) -> dict | None:
    match = re.search(r"\[(white|hispanic|black|asian|others?)\]", text, re.IGNORECASE)
    if not match:
        return None
    race = match.group(1).lower()
    if race == "other":
        race = "others"
    return {"race": race, "prompt": text.strip()}


def parse_height(text: str) -> dict | None:
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None
    height = int(numbers[0])
    if 100 <= height <= 250:
        return {"height": height}
    return None


def parse_color(text: str) -> dict | None:
    lower = text.lower()
    for color in ["black", "white", "red", "blue", "green", "yellow", "brown"]:
        if color in lower:
            return {"color": color}
    return None


def run_parallel(task: str, model_name: str, n_samples: int, temperature: float, max_workers: int, max_tokens: int, parser: Callable[[str], dict | None]) -> List[dict]:
    system_prompt, user_prompt = load_prompt_pair(task)
    results: List[dict] = []

    def worker(_: int) -> dict | None:
        text = call_downstream(model_name, system_prompt, user_prompt, temperature, max_tokens)
        return parser(text)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(n_samples)]
        for future in as_completed(futures):
            item = future.result()
            if item is not None:
                results.append(item)
    return results


def task_defaults(task_name: str) -> tuple[int, int]:
    if task_name == "mcq":
        return DOWNSTREAM_CONFIG["mcq"]["n_samples"], 2048
    if task_name == "joint_attribute":
        return DOWNSTREAM_CONFIG["joint_attributes"]["n_samples"], 1024
    return DOWNSTREAM_CONFIG["independent_attributes"]["n_samples"], 256


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generation runner for downstream tasks in the paper.")
    parser.add_argument("--task", choices=["mcq", "joint_attribute", "independent_gender", "independent_race", "independent_height", "independent_color"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/downstream"))
    args = parser.parse_args()

    default_n, max_tokens = task_defaults(args.task)
    n_samples = args.n_samples or default_n

    task_map: Dict[str, tuple[str, Callable[[str], dict | None], str]] = {
        "mcq": ("mcq", parse_mcq, "all_questions.json"),
        "joint_attribute": ("joint_attribute", parse_joint, "all_results.json"),
        "independent_gender": ("independent_gender", parse_gender, "all_prompts.json"),
        "independent_race": ("independent_race", parse_race, "all_prompts.json"),
        "independent_height": ("independent_height", parse_height, "all_heights.json"),
        "independent_color": ("independent_color", parse_color, "all_colors.json"),
    }
    prompt_key, parser_fn, output_name = task_map[args.task]
    results = run_parallel(prompt_key, args.model, n_samples, args.temperature, args.max_workers, max_tokens, parser_fn)

    out_dir = args.output_dir / args.task / canonical_model_name(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "num_valid": len(results)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
