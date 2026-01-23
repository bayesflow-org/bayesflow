# BayesFlow LLM Context

This folder contains Markdown context files to improve LLM assistance for BayesFlow.

## Files
- `bayesflow-context-compact-<TAG>.md`
  Compact snapshot: README + examples; ~ 50k tokens.
  Recommended for applied use cases that are well-covered in the example tutorial notebooks.

- `bayesflow-context-full-<TAG>.md`
  Full snapshot: README + examples + source code (`bayesflow/`); ~ 250k tokens.
  Recommended for custom use cases that require latest source code references.

## Usage
1. Download either the compact or full file for the current release tag: The compact file is more light-weight and focused; the full file contains the complete codebase.
2. Paste it into your LLM of choice as a context file before asking questions about BayesFlow.

## Prompt Tip
### Compact File
You are answering questions about BayesFlow using the provided context .md file containing all BayesFlow tutorials. If needed, additionally look up the latest source code from the BayesFlow documentation.
QUESTION: <user question here>

### Full File
You are answering questions about BayesFlow using only the provided context .md file containing all BayesFlow tutorials as well as the BayesFlow source code.
QUESTION: <user question here>

## Disclaimer
The context files are generated automatically and may be outdated or incomplete. While they aim at improving LLM accuracy, hallucinations may still occur frequently during LLM assistance. Please always refer to the official BayesFlow documentation and codebase for the most accurate information.

## For BayesFlow Developers
The context files are automatically updated upon new BayesFlow releases by `.github/workflows/build-llm-context.yaml`. The script `llm_context/build_llm_context.py` can also be run manually with an optional `--tag <TAG>` argument (default: `dev`):
```bash
pip install -r llm_context/requirements.txt
python llm_context/build_llm_context.py --tag <TAG>
```
