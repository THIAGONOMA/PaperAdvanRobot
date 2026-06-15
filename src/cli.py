"""CLI do pipeline (Typer).

Exemplos:
    python -m src.cli run --dataset mosei --condition C1 --k-shots 3
    python -m src.cli run --dataset omg --condition C2
"""
from __future__ import annotations

import typer

from .config import CFG

app = typer.Typer(add_completion=False, help="Comparativo emoção: baseline vs LLM.")


@app.command()
def run(
    dataset: str = typer.Option(..., help="omg | mosei"),
    condition: str = typer.Option("C1", help="C1 | C2 | C3"),
    split: str = typer.Option("test", help="split a avaliar"),
    k_shots: int = typer.Option(0, help="número de exemplos few-shot"),
) -> None:
    """Roda a inferência do LLM sobre o split, para todas as seeds configuradas."""
    from .data import mosei_loader, omg_loader
    from .llm.runner import run as run_inference

    loader = {"omg": omg_loader, "mosei": mosei_loader}[dataset]
    for seed in CFG.run.seeds:
        samples = list(loader.iter_samples(split))
        run_inference(samples, dataset=dataset, condition=condition,
                      seed=seed, k_shots=k_shots)
        typer.echo(f"[ok] {dataset}/{condition} seed={seed} ({len(samples)} amostras)")


if __name__ == "__main__":
    app()
