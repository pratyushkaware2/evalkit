from abc import ABC, abstractmethod
from typing import Literal
from evalkit.schema import EvalSample
from datasets import load_dataset

class DatasetAdapter(ABC):
    @abstractmethod
    def normalized(self, 
                   source_split: Literal["test", "train", "validation"]
                   ) -> list[EvalSample]:
       ... 

class GSM8KAdapter(DatasetAdapter):
    def normalized(self, source_split: Literal["test", "train", "validation"]) -> list[EvalSample]:
        raw_gsm8k = load_dataset("openai/gsm8k", "main", split=source_split)

        normalized_dataset = []
        for i, row in enumerate(raw_gsm8k):  
            normalized_dataset.append(EvalSample(id="gsm8k"+str(i),
                                                 dataset="gsm8k",
                                                 prompt=row['question'],
                                                 reference=row['answer'].split("####")[-1].strip(),
                                                 task_type="math",
                                                 source_split=source_split))
        return normalized_dataset



class HumanEvalAdapter(DatasetAdapter):
    def normalized(self, source_split: Literal["test", "train", "validation"]) -> list[EvalSample]:
        if source_split not in ["test"]:
            return []
        raw_humaneval = load_dataset("openai/openai_humaneval", "openai_humaneval", split=source_split)

        normalized_dataset = []
        for row in raw_humaneval:
            normalized_dataset.append(EvalSample(id=row["task_id"],
                                                 dataset="humaneval",
                                                 prompt=row['prompt'],
                                                 reference=row['test'] + "\ncheck(" + row['entry_point'] + ")",
                                                 task_type="code",
                                                 source_split=source_split))
        return normalized_dataset
class MBPPAdapter(DatasetAdapter):
    def normalized(self, source_split: Literal["test", "train", "validation"]) -> list[EvalSample]:
        raw_mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split=source_split)

        normalized_dataset = []
        for row in raw_mbpp:
            normalized_dataset.append(EvalSample(id=str(row["task_id"]),
                                                 dataset="mbpp",
                                                 prompt=row['prompt'],
                                                 reference="\n".join(row["test_imports"]) + "\n" + "\n".join(row['test_list']), 
                                                 task_type="code",
                                                 source_split=source_split))
        return normalized_dataset
