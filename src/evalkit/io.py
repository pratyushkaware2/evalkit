from pyarrow import Table, parquet
from evalkit.schema import EvalSample

def write_parquet(dataset: list[EvalSample], filepath: str) -> None:
    pylist_pydict = [s.model_dump() for s in dataset]
    table = Table.from_pylist(pylist_pydict) 
    parquet.write_table(table, filepath )
    
