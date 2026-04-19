from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from common import REPO_ROOT, compute_overall_pass_rates, compute_protocol_results, format_w1_value, paper_model_order

EXPECTED_BATCH = {
    'GPT-5.2': 13,
    'Gemini-3': 13,
    'GPT-4o': 40,
    'DeepSeek-V3.2': 7,
    'Qwen3': 0,
    'Gemma-3': 7,
    'Mistral-3.2': 0,
    'Kimi-K2': 20,
    'Llama-3.3': 0,
    'Llama-4': 7,
    'GPT-OSS': 13,
}
EXPECTED_INDEPENDENT = {
    'GPT-5.2': 0,
    'Gemini-3': 0,
    'GPT-4o': 0,
    'DeepSeek-V3.2': 0,
    'Qwen3': 0,
    'Gemma-3': 0,
    'Mistral-3.2': 0,
    'Kimi-K2': 0,
    'Llama-3.3': 0,
    'Llama-4': 7,
    'GPT-OSS': 0,
}

SECRET_PATTERNS = [
    re.compile(r'sk-proj-[A-Za-z0-9_-]+'),
    re.compile(r'tgp_v1_[A-Za-z0-9_-]+'),
    re.compile(r'DEEPINFRA_API_KEY\s*=\s*"[A-Za-z0-9]+"'),
    re.compile(r'OPENAI_API_KEY\s*=\s*"sk-'),
]


def fail(msg: str) -> None:
    print(f'ERROR: {msg}')
    sys.exit(1)


def check_no_secrets() -> None:
    for path in REPO_ROOT.rglob('*'):
        if path.is_dir() or path.suffix.lower() in {'.npz', '.pdf', '.png', '.jsonl'}:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except Exception:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                fail(f'Secret-like token found in {path.relative_to(REPO_ROOT)}')


def check_prompt_counts() -> None:
    batch = sorted((REPO_ROOT / 'prompts' / 'batch').glob('*.txt'))
    independent = sorted((REPO_ROOT / 'prompts' / 'independent').glob('*.txt'))
    downstream = sorted((REPO_ROOT / 'prompts' / 'downstream').glob('*.txt'))
    if len(batch) != 15 or len(independent) != 15:
        fail(f'Expected 15 batch and 15 independent prompts, found {len(batch)} and {len(independent)}')
    if len(downstream) != 12:
        fail(f'Expected 12 downstream prompt files, found {len(downstream)}')


def check_required_paths() -> None:
    required = [
        REPO_ROOT / 'README.md',
        REPO_ROOT / 'configs' / 'distributions.json',
        REPO_ROOT / 'data' / 'raw_results' / 'batch',
        REPO_ROOT / 'data' / 'raw_results' / 'independent',
        REPO_ROOT / 'data' / 'reference' / 'reference_samples_aligned_seed42_n1000.npz',
        REPO_ROOT / 'data' / 'processed' / 'recomputed_main_results.json',
        REPO_ROOT / 'data' / 'processed' / 'downstream' / 'downstream_results.json',
        REPO_ROOT / 'data' / 'downstream' / 'mcq' / 'raw_outputs' / 'gpt4o' / 'all_questions.json',
        REPO_ROOT / 'data' / 'downstream' / 'attributes' / 'joint' / 'gpt4o' / 'all_results.json',
        REPO_ROOT / 'data' / 'downstream' / 'attributes' / 'independent' / 'gender' / 'deepseek' / 'all_prompts.json',
        REPO_ROOT / 'src' / 'generate_samples.py',
        REPO_ROOT / 'src' / 'generate_downstream.py',
        REPO_ROOT / 'src' / 'verify_release.py',
        REPO_ROOT / 'scripts' / 'validate_release.sh',
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        fail(f'Missing required paths: {missing}')



def pct(passed: int, total: int) -> int:
    return round(100 * passed / total) if total else 0


def check_pass_rates() -> None:
    batch = compute_protocol_results('batch')
    independent = compute_protocol_results('independent')
    batch_rates = compute_overall_pass_rates(batch, alpha=0.01)
    ind_rates = compute_overall_pass_rates(independent, alpha=0.01)
    for model in paper_model_order():
        got = pct(*batch_rates[model])
        if got != EXPECTED_BATCH[model]:
            fail(f'Batch pass rate mismatch for {model}: got {got}, expected {EXPECTED_BATCH[model]}')
        got = pct(*ind_rates[model])
        if got != EXPECTED_INDEPENDENT[model]:
            fail(f'Independent pass rate mismatch for {model}: got {got}, expected {EXPECTED_INDEPENDENT[model]}')

    if format_w1_value(batch['Gaussian']['GPT-4o']['w1']) != '0.10':
        fail('Unexpected batch Gaussian / GPT-4o W1')
    if format_w1_value(batch['Uniform']['DeepSeek-V3.2']['w1']) != '9e-03':
        fail('Unexpected batch Uniform / DeepSeek-V3.2 W1')
    if format_w1_value(independent['Uniform']['GPT-4o']['w1']) != '0.16':
        fail('Unexpected independent Uniform / GPT-4o W1')
    if format_w1_value(independent['Bernoulli']['Llama-4']['w1']) != '0.02':
        fail('Unexpected independent Bernoulli / Llama-4 W1')


def check_processed_json() -> None:
    main_json = json.loads((REPO_ROOT / 'data' / 'processed' / 'recomputed_main_results.json').read_text(encoding='utf-8'))
    downstream_json = json.loads((REPO_ROOT / 'data' / 'processed' / 'downstream' / 'downstream_results.json').read_text(encoding='utf-8'))
    if sorted(main_json.keys()) != ['batch', 'independent']:
        fail('Unexpected top-level keys in recomputed_main_results.json')
    if sorted(downstream_json.keys()) != ['independent_attributes', 'joint_attributes', 'mcq']:
        fail('Unexpected top-level keys in downstream_results.json')



def main() -> None:
    check_required_paths()
    check_prompt_counts()
    check_no_secrets()
    check_pass_rates()
    check_processed_json()
    print('verify_release.py: all checks passed')


if __name__ == '__main__':
    main()
