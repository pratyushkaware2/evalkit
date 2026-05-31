from pydantic import BaseModel
from typing import Literal

class EvalSample(BaseModel):
    id: str
    dataset: Literal["gsm8k", "humaneval", "mbpp"]
    prompt: str
    reference: str
    task_type: Literal["math", "code"]
    source_split: Literal["test", "train", "validation"] 
