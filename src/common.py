from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
PROMPT_DIR = REPO_ROOT / "prompts"
TABLE_DIR = REPO_ROOT / "tables"
FIGURE_DIR = REPO_ROOT / "figures"
MANIFEST_DIR = REPO_ROOT / "manifests"

MODEL_DISPLAY_MAP = {
    "gpt-4o": "GPT-4o",
    "gpt-5.2": "GPT-5.2",
    "gpt_5.2": "GPT-5.2",
    "deepseek-ai/DeepSeek-V3.2-Exp": "DeepSeek-V3.2",
    "deepseek-ai_DeepSeek-V3.2-Exp": "DeepSeek-V3.2",
    "Qwen/Qwen3-32B": "Qwen3",
    "Qwen_Qwen3-32B": "Qwen3",
    "google/gemma-3-27b-it": "Gemma-3",
    "google_gemma-3-27b-it": "Gemma-3",
    "gemini-3-pro-preview": "Gemini-3",
    "gemini_3_pro_preview": "Gemini-3",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "Mistral-3.2",
    "mistralai_Mistral-Small-3.2-24B-Instruct-2506": "Mistral-3.2",
    "moonshotai/Kimi-K2-Instruct-0905": "Kimi-K2",
    "moonshotai_Kimi-K2-Instruct-0905": "Kimi-K2",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-3.3",
    "meta-llama_Llama-3.3-70B-Instruct-Turbo": "Llama-3.3",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3",
    "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama-4",
    "meta-llama_Llama-4-Scout-17B-16E-Instruct": "Llama-4",
    "openai/gpt-oss-120b": "GPT-OSS",
    "openai_gpt-oss-120b": "GPT-OSS",
}

LATEX_DIST_NAMES = {
    "Bernoulli": "Bernoulli",
    "Binomial": "Binomial",
    "Poisson": "Poisson",
    "Uniform": "Uniform",
    "Gaussian": "Gaussian",
    "Beta": "Beta",
    "Exponential": "Exp",
    "Cauchy": "Cauchy",
    "TDistribution": "$t$",
    "ChiSquare": "$\\chi^2$",
    "FDistribution": "$F$",
    "Gamma": "Gamma",
    "Weibull": "Weibull",
    "Laplace": "Laplace",
    "Logistic": "Logistic",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_configs() -> Tuple[dict, dict, dict, dict]:
    distributions = load_json(CONFIG_DIR / "distributions.json")
    protocols = load_json(CONFIG_DIR / "protocols.json")
    evaluation = load_json(CONFIG_DIR / "evaluation.json")
    models = load_json(CONFIG_DIR / "models.template.json")
    return distributions, protocols, evaluation, models


def paper_model_order() -> List[str]:
    _, _, _, models = load_configs()
    return models["model_order"]


def display_name_from_raw(raw_name: str) -> str:
    if raw_name in MODEL_DISPLAY_MAP:
        return MODEL_DISPLAY_MAP[raw_name]
    normalized = raw_name.replace("/", "_")
    if normalized in MODEL_DISPLAY_MAP:
        return MODEL_DISPLAY_MAP[normalized]
    return raw_name


def find_candidate_files(result_dir: Path) -> List[Path]:
    files = sorted(result_dir.glob("*.json"))
    return [p for p in files if "backup" not in p.name.lower() and "retest" not in p.name.lower()]


def load_reference_npz() -> np.lib.npyio.NpzFile:
    return np.load(DATA_DIR / "reference" / "reference_samples_aligned_seed42_n1000.npz")


def reference_samples(protocol: str, distribution: str) -> np.ndarray:
    npz = load_reference_npz()
    return np.array(npz[f"{protocol}/{distribution}"])


def extract_model_name(payload: dict) -> str:
    return payload.get("model") or payload.get("model_name") or "unknown"


def load_protocol_distribution_results(protocol: str, distribution: str) -> Dict[str, dict]:
    result_dir = DATA_DIR / "raw_results" / protocol / distribution
    results = {}
    if not result_dir.exists():
        return results
    for path in find_candidate_files(result_dir):
        payload = load_json(path)
        display_name = display_name_from_raw(extract_model_name(payload))
        if display_name in results:
            continue
        samples = payload.get("samples", [])
        if not isinstance(samples, list) or len(samples) < 10:
            continue
        results[display_name] = {
            "payload": payload,
            "path": path,
            "samples": np.array(samples, dtype=float)
        }
    return results


def compute_chisquare_bernoulli(samples: Iterable[float], p: float = 0.7) -> Tuple[float, float]:
    values = list(samples)
    n = len(values)
    count_0 = sum(1 for x in values if x in (0, 0.0))
    count_1 = sum(1 for x in values if x in (1, 1.0))
    observed = np.array([count_0, count_1])
    expected = np.array([n * (1 - p), n * p])
    stat, pval = stats.chisquare(observed, f_exp=expected)
    return float(stat), float(pval)


def compute_chisquare_binomial(samples: Iterable[float], n_trials: int = 10, p: float = 0.5) -> Tuple[float, float]:
    values = [int(x) for x in samples]
    n = len(values)
    counter = Counter(values)
    observed = np.array([counter.get(k, 0) for k in range(n_trials + 1)])
    expected = np.array([stats.binom.pmf(k, n_trials, p) * n for k in range(n_trials + 1)])
    while len(expected) > 2 and (expected[0] < 5 or expected[-1] < 5):
        if expected[0] < 5:
            observed = np.concatenate([[observed[0] + observed[1]], observed[2:]])
            expected = np.concatenate([[expected[0] + expected[1]], expected[2:]])
        if len(expected) > 2 and expected[-1] < 5:
            observed = np.concatenate([observed[:-2], [observed[-2] + observed[-1]]])
            expected = np.concatenate([expected[:-2], [expected[-2] + expected[-1]]])
    stat, pval = stats.chisquare(observed, f_exp=expected)
    return float(stat), float(pval)


def compute_chisquare_poisson(samples: Iterable[float], lam: float = 5.0, max_k: int = 15) -> Tuple[float, float]:
    values = [int(x) for x in samples]
    n = len(values)
    counter = Counter(values)
    observed = []
    expected = []
    for k in range(max_k):
        observed.append(counter.get(k, 0))
        expected.append(stats.poisson.pmf(k, lam) * n)
    observed.append(sum(counter.get(k, 0) for k in counter if k >= max_k))
    expected.append((1 - stats.poisson.cdf(max_k - 1, lam)) * n)
    observed = np.array(observed)
    expected = np.array(expected)
    while len(expected) > 2 and (expected[0] < 5 or expected[-1] < 5):
        if expected[-1] < 5:
            observed = np.concatenate([observed[:-2], [observed[-2] + observed[-1]]])
            expected = np.concatenate([expected[:-2], [expected[-2] + expected[-1]]])
        if len(expected) > 2 and expected[0] < 5:
            observed = np.concatenate([[observed[0] + observed[1]], observed[2:]])
            expected = np.concatenate([[expected[0] + expected[1]], expected[2:]])
    stat, pval = stats.chisquare(observed, f_exp=expected)
    return float(stat), float(pval)


def compute_metrics(distribution: str, samples: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    if distribution == "Bernoulli":
        _, pval = compute_chisquare_bernoulli(samples, p=0.7)
    elif distribution == "Binomial":
        _, pval = compute_chisquare_binomial(samples, n_trials=10, p=0.5)
    elif distribution == "Poisson":
        _, pval = compute_chisquare_poisson(samples, lam=5)
    else:
        _, pval = stats.ks_2samp(samples, ref)
    w1 = float(stats.wasserstein_distance(samples, ref))
    return float(pval), w1


def paper_distribution_order() -> List[str]:
    distributions, _, _, _ = load_configs()
    return distributions["paper_distribution_order"]


def compute_protocol_results(protocol: str) -> Dict[str, Dict[str, dict]]:
    results: Dict[str, Dict[str, dict]] = {}
    for distribution in paper_distribution_order():
        dist_results = {}
        ref = reference_samples(protocol, distribution)
        for display_name, item in load_protocol_distribution_results(protocol, distribution).items():
            pval, w1 = compute_metrics(distribution, item["samples"], ref)
            dist_results[display_name] = {
                "pval": pval,
                "w1": w1,
                "source_file": str(item["path"].relative_to(REPO_ROOT)),
                "num_samples": int(len(item["samples"]))
            }
        results[distribution] = dist_results
    return results


def format_w1_value(w1: float) -> str:
    if w1 < 0.01:
        return f"{w1:.0e}"
    if w1 < 1:
        return f"{w1:.2f}"
    return f"{w1:.1f}"


def format_main_cell(w1: float, pval: float, alpha: float) -> str:
    value = format_w1_value(w1)
    if pval > alpha:
        return f"\\textbf{{{value}}}$^*$"
    return value


def format_pass_rate(passed: int, total: int) -> str:
    if total == 0:
        return "-"
    pct = round(100 * passed / total)
    if pct > 0:
        return f"\\textbf{{{pct}\\%}}"
    return f"{pct}\\%"


def tier_groups() -> Dict[str, List[str]]:
    distributions, _, _, _ = load_configs()
    groups: Dict[str, List[str]] = {"Tier I": [], "Tier II": [], "Tier III": []}
    for name, meta in distributions["distributions"].items():
        groups[meta["tier"]].append(name)
    return groups


def compute_tier_summary(protocol_results: Dict[str, Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
    groups = tier_groups()
    model_order = paper_model_order()
    summary: Dict[str, Dict[str, dict]] = {}
    for tier_name, dists in groups.items():
        tier_stats = {}
        for model in model_order:
            entries = [protocol_results[d].get(model) for d in dists if protocol_results.get(d, {}).get(model)]
            if not entries:
                tier_stats[model] = None
                continue
            passed = sum(1 for item in entries if item["pval"] > 0.01)
            mean_w1 = sum(item["w1"] for item in entries) / len(entries)
            tier_stats[model] = {"passed": passed, "total": len(entries), "mean_w1": mean_w1}
        summary[tier_name] = tier_stats
    return summary


def compute_overall_pass_rates(protocol_results: Dict[str, Dict[str, dict]], alpha: float = 0.01) -> Dict[str, Tuple[int, int]]:
    counts = {model: [0, 0] for model in paper_model_order()}
    for distribution in paper_distribution_order():
        for model in paper_model_order():
            item = protocol_results.get(distribution, {}).get(model)
            if item is None:
                continue
            counts[model][1] += 1
            if item["pval"] > alpha:
                counts[model][0] += 1
    return {model: (vals[0], vals[1]) for model, vals in counts.items()}


def sanitize_number_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def parse_all_numbers(text: str) -> List[float]:
    cleaned = sanitize_number_text(text)
    matches = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", cleaned)
    out = []
    for match in matches:
        try:
            out.append(float(match))
        except ValueError:
            continue
    return out


def parse_single_number(text: str) -> float | None:
    values = parse_all_numbers(text)
    return values[0] if values else None


def load_prompt(protocol: str, distribution: str) -> str:
    return (PROMPT_DIR / protocol / f"{distribution}.txt").read_text(encoding="utf-8")


def prompt_format_kwargs(distribution: str, n_samples: int | None = None) -> dict:
    distributions, _, _, _ = load_configs()
    meta = distributions["distributions"][distribution]["parameters"].copy()
    if n_samples is not None:
        meta["n_samples"] = n_samples
    if distribution == "ChiSquare":
        meta.setdefault("k", meta["df"])
    if distribution == "FDistribution":
        meta.setdefault("d1", meta["dfn"])
        meta.setdefault("d2", meta["dfd"])
    if distribution == "Gamma":
        meta.setdefault("k", meta["shape"])
    if distribution == "Weibull":
        meta.setdefault("k", meta["shape"])
        meta.setdefault("lambda", meta["scale"])
    if distribution == "Laplace":
        meta.setdefault("mu", meta["loc"])
        meta.setdefault("b", meta["scale"])
    if distribution == "Logistic":
        meta.setdefault("mu", meta["loc"])
        meta.setdefault("s", meta["scale"])
    if distribution == "Cauchy":
        meta.setdefault("x0", meta["loc"])
        meta.setdefault("gamma", meta["scale"])
    return meta
