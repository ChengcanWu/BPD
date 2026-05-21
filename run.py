"""Entry point: run flat or hierarchy MAS, apply BPD, repair, and save results."""

import os
import pickle
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from mas.llm import init_llm
from mas.runner import communicate_with_bpd


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    load_dotenv()
    cfg = load_config()

    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    init_llm(
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", cfg.get("model", "deepseek-chat")),
        record_path=cfg.get("record_path", "outputs/run.log"),
    )

    df = pd.read_parquet(cfg["data_path"])
    start = int(cfg.get("question_start", 0))
    end = int(cfg.get("question_end", 1))
    all_results = []

    structure = cfg.get("mas_structure", "hierarchy")
    if structure not in ("flat", "hierarchy"):
        raise ValueError("mas_structure must be 'flat' or 'hierarchy'")

    for question_id in range(start, end):
        print(f"Question {question_id} [{structure}] ...")
        result = communicate_with_bpd(
            question_id,
            df,
            structure=structure,
            epsilon=float(cfg.get("epsilon", 1.5)),
            malicious_agent_id=int(cfg.get("malicious_agent_id", 1)),
            num_respondents=int(cfg.get("num_respondents", 5)),
            num_advisors=int(cfg.get("num_advisors", 2)),
        )
        detected = result["detected_agent"]
        print(f"  BPD detected agent: {detected} (deviation={result['deviation']:.3f})")
        print(f"  MAS final choice (before repair): {result['final_choice']}")

        out_path = Path(cfg["results_dir"]) / f"result_{question_id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        all_results.append(result)

    summary_path = Path(cfg["results_dir"]) / "all_results.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved results to {cfg['results_dir']}")


if __name__ == "__main__":
    main()
