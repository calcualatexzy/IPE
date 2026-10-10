"""Token statistics and a histogram for a dataset.jsonl (port of upstream src/utils/training_data/count_tokens.py)."""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def count_dataset_tokens(dataset_path: Path, tokenizer_name: str, plot_path: Path, exact: bool = False,
                         sample_size: int = 500, seed: int = 0) -> dict:
    """Count the tokens of every document's text, or estimate the total from a sample, and save a histogram."""
    with open(dataset_path, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    if not texts:
        return {"n_documents": 0, "total_tokens": 0, "tokens_per_document": 0, "max_token_length": 0}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sample = texts if exact else random.Random(seed).sample(texts, min(sample_size, len(texts)))
    counts = [len(tokenizer.encode(text)) for text in sample]
    avg_tokens = sum(counts) / len(counts)
    total_tokens = sum(counts) if exact else int(avg_tokens * len(texts))

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Token Count Distribution: {dataset_path.parent.name}\n"
              f"({'Exact' if exact else f'Sample Size: {len(counts)}'})")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.5)
    plt.axvline(avg_tokens, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {avg_tokens:.1f}')
    plt.axvline(np.median(counts), color='g', linestyle='dashed', linewidth=1, label=f'Median: {np.median(counts):.1f}')
    plt.axvline(min(counts), color='orange', linestyle='dashed', linewidth=1, label=f'Min: {min(counts)}')
    plt.axvline(max(counts), color='b', linestyle='dashed', linewidth=1, label=f'Max: {max(counts)}')
    plt.legend()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    return {
        "n_documents": len(texts),
        "total_tokens": total_tokens,
        "tokens_exact": exact,
        "tokens_per_document": int(avg_tokens),
        "max_token_length": max(counts),  # of the sample when not exact
    }
