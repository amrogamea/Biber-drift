# Paraphrase test on the HPC

The laptop drew the sample (`biber-drift paraphrase sample`). The HPC paraphrases it with
open-weight models, one GPU job per model (vLLM), and the analysis runs on the HPC too. All
settings are in [`configs/paraphrase.yaml`](../configs/paraphrase.yaml).

The sample and the outputs contain AO3 chapters and ACL papers and paraphrases of them. Keep them
in private storage (`$HOME` and `$WORK` are private here). Never commit or share them.

## The HPC: NHR@FAU TinyGPU

- Login node `tinyx`. Slurm commands go through the cluster wrappers: `sbatch.tinygpu`,
  `salloc.tinygpu`, `squeue.tinygpu`, `scancel.tinygpu` (they add `-M tinygpu`). The account is the
  default one (`iwso`); Slurm sets no GPU-hour cap.
- GPUs: only partition `a100` fits: 8 nodes with 4x A100 **40 GB**. Every GPU comes with 32 cores
  and 117 GB of RAM, so jobs ask for no cores or memory. The RTX 3080 / 2080 Ti cards (`work`,
  `rtx3080`) have 10–11 GB, too little for an 8B model; vLLM does not run on the V100s.
- Limits: batch jobs 24 h, interactive jobs (`salloc.tinygpu`) 4 h.
- Storage: `$HOME` (~105 GB quota) holds the code, data and outputs; `$WORK` (the work file
  system, 1 TB) holds the model weights (~150 GB) and the vLLM environment.
- The system `python3.12` cannot create a venv (no `ensurepip`); use the cluster's conda Python
  `/apps/python/3.12-conda/bin/python3.12`.
- `pip install vllm==0.32.0` brings torch built for CUDA 13.0, which needs an NVIDIA driver of 580
  or newer. The GPU nodes have driver 615.71.10 (CUDA up to 13.4, checked 2026-11-04), so it works.
  With an older driver, install vLLM's cu128 build instead. Compute nodes have no internet: download
  on the login node.

## 1. Once: environment variables, environment, Hugging Face access

Put these lines in `~/.bashrc` (and make `~/.bash_profile` source `~/.bashrc`, so SSH logins read
them too):

```bash
export HF_HOME=$WORK/hf-cache            # model weights + the token from `hf auth login`
export VLLM_ENV=$WORK/envs/vllm          # used by setup_vllm_env.sh and generate.sbatch
export PIP_CACHE_DIR=$WORK/.cache/pip
```

Then, on the login node, from the repository:

```bash
PYTHON=/apps/python/3.12-conda/bin/python3.12 bash hpc/setup_vllm_env.sh
source "$VLLM_ENV/bin/activate"
hf auth login                         # paste a read token; it is stored under $HF_HOME, never in code
```

Llama and Gemma are gated. First accept the licences of `meta-llama/Llama-3.1-8B-Instruct`,
`meta-llama/Llama-2-13b-chat-hf` and `google/gemma-3-12b-it` on huggingface.co, with the account
the token belongs to.

## 2. Download the weights (login node)

Run it inside `tmux`, so a dropped SSH connection does not stop it:

```bash
biber-drift paraphrase download                       # every model in the config, ~150 GB
biber-drift paraphrase download --model qwen3-8b      # or some of them (comma-separated)
```

The log shows each model's commit and size. The GPU jobs run offline from this cache
(`HF_HUB_OFFLINE=1`).

## 3. Pilot (400 texts per model)

A quick test first: every model on 20 texts per corpus (`--limit` spreads them over the groups
and years), into a throwaway folder (one array task per model):

```bash
mkdir -p logs
sbatch.tinygpu --array=0-6 --time=00:30:00 hpc/generate.sbatch pilot --limit 20 --name smoke
```

Check it with the same selection (otherwise the other 360 pilot texts count as missing):

```bash
biber-drift paraphrase check --stage pilot --name smoke --limit 20
```

Then the pilot, all seven models (the Llama 2 check included):

```bash
sbatch.tinygpu --array=0-6 --time=01:00:00 hpc/generate.sbatch pilot
```

Each model writes `outputs/paraphrase/generations/pilot/<model>.jsonl` and `<model>.meta.json`.
The meta file records the settings, the commit, the vLLM/torch versions, the GPU and the
throughput. A task that stops resumes where it ended when submitted again. Check the outputs before
the main run (refusals, chatty openers, length ratio, loops):

```bash
biber-drift paraphrase check --stage pilot
```

## 4. Main run (15,357 texts per model)

On the RTX 3080 nodes (as run from 2026-11-05), every model split over 4 cards. The variables must
be set in the submitting shell (`--export=ALL` passes them on):

```bash
sbatch.tinygpu --export=ALL --array=0-6 --partition=rtx3080 --gres=gpu:rtx3080:4 --time=16:00:00 --job-name=paraphrase-rtx hpc/generate.sbatch main --tensor-parallel 4
```

The 10 GB cards leave KV cache for only a few texts at once for OLMo 2 (4,300 tokens), Llama 2
(7,400) and Gemma 3 (9,000), so these run at 0.4–0.8 texts/s, about 5–11 hours each. A task that
reaches its time limit resumes when its index is submitted again, e.g. `--array=5`.

On an A100:

```bash
sbatch.tinygpu hpc/generate.sbatch main
sbatch.tinygpu --array=6 hpc/generate.sbatch main     # Llama 2 check
```

On a 40 GB A100, OLMo 2 13B and Llama 2 13B are the slow ones: they use full multi-head attention
(about 0.8 MB of KV cache per token), so only a few texts fit on the GPU at once. Two GPUs each
(`tensor_parallel_size: 2` in the config, `--gres=gpu:a100:2` for that task) should be much faster.
The pilot's meta files give the real texts per second, so `--time` can be set from them. A resumed
run refuses to mix settings: change the prompt or decoding only with a new output folder, e.g. the
temperature check on the pilot texts:
`biber-drift paraphrase generate --model 1 --stage pilot --name pilot_t07 --set generation.temperature=0.7`.

## 5. Clean, filter and tag (`biber-drift paraphrase tag`)

The rules are frozen in the `filters` section of the config; the tagging settings in `tagging`.
Once, on the login node, the tagging environment (spaCy `en_core_web_trf`, pybiber; about 10 GB):

```bash
PYTHON=/apps/python/3.12-conda/bin/python3.12 TAG_ENV=$WORK/envs/tag bash hpc/setup_tag_env.sh
```

Then one GPU job per key: task 0 re-tags the sample's originals, task i >= 1 the model at index
i-1. One RTX 3080 per task is enough:

```bash
sbatch.tinygpu --export=ALL hpc/tag.sbatch main
```

Each task writes `outputs/paraphrase/features/main/<key>.parquet` (Biber features by `item_id`),
`<key>.items.parquet` (kept or not, why, cleaners applied, word counts; no text) and `<key>.json`
(settings, package versions, device). A finished key is skipped when submitted again, and a stopped
one resumes from its tagging cache.
