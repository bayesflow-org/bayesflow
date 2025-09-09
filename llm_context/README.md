# BayesFlow LLM Context

This folder contains single-file context artifacts to improve LLM assistance for BayesFlow.

## Files
- `bayesflow-context-full-<TAG>.md`
  Full Markdown snapshot: README, examples (converted to Markdown), and all `bayesflow/` code.

- `bayesflow-context-compact-<TAG>.md`
  Smaller snapshot: README + examples fully, `bayesflow/` code partially (truncated previews).

- `bayesflow-context-<TAG>.manifest.json`
  Metadata (tag, commit, dependencies, file sizes).

## Usage
1. Download either the full or compact file for the release tag of interest: The compact file is cheaper and faster; the full file is most accurate.
2. Paste it into your LLM context before asking questions about BayesFlow.

## Prompt Tip
You are answering questions about BayesFlow using only the provided context .md file. If using code, cite the file or notebook name shown in the context.
QUESTION: <user question here>

## Disclaimer
The context files are generated automatically and may be outdated or incomplete. While they aim at improving LLM accuracy, hallucinations may still occur frequently during LLM assistance. Please always refer to the official BayesFlow documentation and codebase for the most accurate information.
