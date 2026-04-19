# Large Language Models Are Bad Dice Players

Code, prompts, and released data for the ACL Main 2026 paper:

**Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions**

## File structure

```text
configs/
  distributions.json        distribution definitions used in the main benchmark
  protocols.json            batch / independent protocol settings
  evaluation.json           evaluation settings
  models.template.json      model registry without secrets
  downstream_tasks.json     downstream task settings

prompts/
  batch/                    batch prompts for the main benchmark
  independent/              independent prompts for the main benchmark
  downstream/               prompts for MCQ and downstream attribute tasks

data/
  raw_results/
    batch/                  released main benchmark batch outputs
    independent/            released main benchmark independent outputs
    reference/              reference samples used for evaluation
  downstream/
    mcq/raw_outputs/        released MCQ outputs
    attributes/joint/       released joint attribute outputs
    attributes/independent/ released single-attribute outputs
  processed/
    recomputed_main_results.json
    downstream/downstream_results.json

src/
  generate_samples.py       generate main benchmark raw outputs
  generate_downstream.py    generate downstream raw outputs
  verify_release.py         release checks

scripts/
  validate_release.sh
```

## Existing data

Released raw data already included in the repository:

- `data/raw_results/batch/`
- `data/raw_results/independent/`
- `data/downstream/mcq/raw_outputs/`
- `data/downstream/attributes/joint/`
- `data/downstream/attributes/independent/`

Processed JSON summaries:

- `data/processed/recomputed_main_results.json`
- `data/processed/downstream/downstream_results.json`

## Re-running generation

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API keys

The generation scripts read credentials from environment variables defined in `configs/models.template.json`.

```bash
export OPENAI_API_KEY=...
export DEEPINFRA_API_KEY=...
export GEMINI_API_KEY=...
```

### 3. Run generation

Main benchmark, batch mode:

```bash
python src/generate_samples.py \
  --protocol batch \
  --model GPT-4o \
  --distribution Gaussian \
  --n-samples 1000 \
  --output-dir outputs/main
```

Main benchmark, independent mode:

```bash
python src/generate_samples.py \
  --protocol independent \
  --model GPT-4o \
  --distribution Uniform \
  --n-samples 1000 \
  --output-dir outputs/main
```

Downstream MCQ:

```bash
python src/generate_downstream.py \
  --task mcq \
  --model GPT-4o \
  --n-samples 1000 \
  --output-dir outputs/downstream
```

Downstream joint attributes:

```bash
python src/generate_downstream.py \
  --task joint_attribute \
  --model DeepSeek-V3.2 \
  --n-samples 1000 \
  --output-dir outputs/downstream
```

Downstream single attribute:

```bash
python src/generate_downstream.py \
  --task independent_height \
  --model GPT-OSS \
  --n-samples 1000 \
  --output-dir outputs/downstream
```

### 4. Output directories

Newly generated raw outputs are written to:

- `outputs/main/`
- `outputs/downstream/`

## Validation

```bash
bash scripts/validate_release.sh
```
