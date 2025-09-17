# BayesFlow LLM Context

This folder contains single-file context artifacts to improve LLM assistance for BayesFlow.

## Files
- `bayesflow-context-compact-<TAG>.md`
  Smaller snapshot: README + examples; ~ 50k tokens.

- `bayesflow-context-full-<TAG>.md`
  Full Markdown snapshot: README + examples + source code (`bayesflow/`); ~ 250k tokens.

## Usage
1. Download either the compact or full file for the release tag of interest: The compact file is cheaper and faster; the full file is most accurate.
2. Paste it into your LLM context before asking questions about BayesFlow.

## Prompt Tip
### Compact File
You are answering questions about BayesFlow using the provided context .md file containing all BayesFlow tutorials. If needed, look up the latest source code from the BayesFlow documentation.
QUESTION: <user question here>

### Full File
You are answering questions about BayesFlow using only the provided context .md file containing all BayesFlow tutorials as well as the BayesFlow source code.
QUESTION: <user question here>

## Disclaimer
The context files are generated automatically and may be outdated or incomplete. While they aim at improving LLM accuracy, hallucinations may still occur frequently during LLM assistance. Please always refer to the official BayesFlow documentation and codebase for the most accurate information.

## For Developers
The context files are automatically updated upon new BayesFlow releases by `.github/workflows/build-llm-context.yaml`. The script `llm_context/build_llm_context.py` can also be run manually with an optional `--tag <TAG>` argument (default: `dev`):
```bash
pip install -r llm_context/requirements.txt
python llm_context/build_llm_context.py --tag <TAG>
```
