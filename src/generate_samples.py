from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from common import CONFIG_DIR, PROMPT_DIR, load_json, parse_all_numbers, parse_single_number, prompt_format_kwargs

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


def load_model_config(model_name: str) -> Dict[str, Any]:
    config = load_json(CONFIG_DIR / "models.template.json")
    return config["models"][model_name]


def build_prompt(protocol: str, distribution: str, n_samples: int) -> str:
    template = (PROMPT_DIR / protocol / f"{distribution}.txt").read_text(encoding="utf-8")
    kwargs = prompt_format_kwargs(distribution, n_samples=n_samples)
    return template.format(**kwargs)


def make_openai_client(model_cfg: Dict[str, Any]):
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    api_key = os.getenv(model_cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing environment variable: {model_cfg['api_key_env']}")
    kwargs = {"api_key": api_key}
    if model_cfg.get("base_url"):
        kwargs["base_url"] = model_cfg["base_url"]
    return OpenAI(**kwargs)


def call_model(model_cfg: Dict[str, Any], prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    provider = model_cfg["provider"]
    if provider == "openai_compatible":
        client = make_openai_client(model_cfg)
        response = client.chat.completions.create(
            model=model_cfg["api_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
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
            contents=prompt,
            config={"temperature": temperature, "top_p": top_p},
        )
        return getattr(response, "text", "") or ""
    raise RuntimeError(f"Unsupported provider: {provider}")


def run_batch(model_name: str, distribution: str, n_samples: int, temperature: float, top_p: float, out_dir: Path) -> Dict[str, Any]:
    model_cfg = load_model_config(model_name)
    prompt = build_prompt("batch", distribution, n_samples)
    text = call_model(model_cfg, prompt, temperature, top_p, max_tokens=20000)
    samples = parse_all_numbers(text)
    payload = {
        "model": model_cfg["api_model"],
        "display_model": model_name,
        "distribution": distribution,
        "protocol": "batch",
        "n_samples_requested": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_file": f"prompts/batch/{distribution}.txt",
        "raw_text": text,
        "samples": samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"batch_{model_name}_{distribution}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"output": str(out_path), "num_samples": len(samples)}


def run_independent(model_name: str, distribution: str, n_samples: int, temperature: float, top_p: float, delay: float, out_dir: Path) -> Dict[str, Any]:
    model_cfg = load_model_config(model_name)
    prompt = build_prompt("independent", distribution, 1)
    raw_texts: List[str] = []
    samples: List[float] = []
    for _ in range(n_samples):
        text = call_model(model_cfg, prompt, temperature, top_p, max_tokens=64)
        raw_texts.append(text)
        value = parse_single_number(text)
        if value is not None:
            samples.append(value)
        if delay > 0:
            time.sleep(delay)
    payload = {
        "model": model_cfg["api_model"],
        "display_model": model_name,
        "distribution": distribution,
        "protocol": "independent",
        "n_requests": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_file": f"prompts/independent/{distribution}.txt",
        "raw_texts": raw_texts,
        "samples": samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"independent_{model_name}_{distribution}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"output": str(out_path), "num_samples": len(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generation runner for the paper's main distribution-sampling experiments.")
    parser.add_argument("--protocol", choices=["batch", "independent"], required=True)
    parser.add_argument("--model", required=True, help="Display model name from configs/models.template.json")
    parser.add_argument("--distribution", required=True, help="Distribution key from configs/distributions.json")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between independent requests in seconds.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.protocol == "batch":
        result = run_batch(args.model, args.distribution, args.n_samples, args.temperature, args.top_p, args.output_dir)
    else:
        result = run_independent(args.model, args.distribution, args.n_samples, args.temperature, args.top_p, args.delay, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
