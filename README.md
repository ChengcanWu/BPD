# Monitoring LLM-Based Multi-Agent Systems (BPD)

Official implementation of **Backward Propagation Detection (BPD)** from:

> *Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation* (ICML 2026)

BPD models MAS communication as a signed DAG, scores edges with an external LLM, detects malicious agents via backward propagation, and repairs the graph by isolating detected attackers.

## Repository layout

```
mas/
  prompts.py   # Agent / advisor / score prompts (Appendix A.2)
  llm.py       # API client and edge scoring
  graph.py     # Signed DAG and multi-round execution
  bpd.py       # Backward propagation, detection, repair helpers
  runner.py    # Flat / hierarchy MAS + full BPD pipeline
run.py         # CLI entry point
config.yaml    # Hyperparameters (epsilon=1.5, etc.)
```

## Requirements

- Python 3.10+
- An OpenAI-compatible API (DeepSeek, GPT-4o, etc.)
- MMLU parquet data (see below)

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key and base URL
```

## Data

Place MMLU test parquet files under `MMLU/`, e.g.:

```
MMLU/college_chemistry/test-00000-of-00001.parquet
```

Download from [Hugging Face MMLU](https://huggingface.co/datasets/cais/mmlu) or mirror. Update `data_path` in `config.yaml` if needed.

## Usage

```bash
python run.py
```

### MAS structures (paper §4.2)

| Structure | Topology | Flow |
|-----------|----------|------|
| **hierarchy** | 5 + 2 + 5 | 5 respondents answer → 2 advisors review each → 5 final summaries |
| **flat** | 5 + 5 + 5 | 5 agents answer → mutual peer suggestions (each reviews others) → 5 final summaries |

Set `mas_structure: flat` or `mas_structure: hierarchy` in `config.yaml`.

Pipeline per question:

1. Run the selected MAS under attack.
2. Build signed edge weights in `{-1, 0, 1}`.
3. **Backward propagate** node scores (Eq. 1) and **detect** malicious agents (Eq. 2, `epsilon=1.5`).
4. **Repair**: re-run MAS with the detected agent switched to benign mode and outgoing edges cleared.

Outputs are saved under `results/` (pickle per question). Logs go to `outputs/run.log`.

### Config highlights (`config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `mas_structure` | hierarchy | `flat` or `hierarchy` |
| `epsilon` | 1.5 | Malicious-agent detection threshold (paper default) |
| `malicious_agent_id` | 1 | Attacked respondent in the harmful attack setting |
| `question_start` / `question_end` | 0 / 1 | Question index range |
| `model` | deepseek-chat | Overridden by `OPENAI_MODEL` in `.env` |

## Citation

```bibtex
@inproceedings{wu2026bpd,
  title={Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation},
  author={Wu, Chengcan and Zhang, Zhixin and Xu, Mingqian and Wei, Zeming and Sun, Meng},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
