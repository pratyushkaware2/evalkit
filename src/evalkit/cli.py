import typer
from typing import Literal
from evalkit.adapters import GSM8KAdapter, HumanEvalAdapter, MBPPAdapter
from evalkit.io import write_parquet
app = typer.Typer()

@app.command()
def normalize(dataset: Literal["gsm8k", "humaneval", "mbpp"] = typer.Option(...), split: Literal["test", "train", "validation"] = typer.Option("test"), output: str = typer.Option("./output.parquet") ) -> None:
    if dataset=="gsm8k":
        write_parquet(GSM8KAdapter().normalized(split), output)
    elif dataset=="humaneval":
        write_parquet(HumanEvalAdapter().normalized(split), output)
    elif dataset=="mbpp":
        write_parquet(MBPPAdapter().normalized(split), output)
    
