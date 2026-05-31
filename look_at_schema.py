from datasets import get_dataset_config_names, load_dataset
print(get_dataset_config_names("openai/gsm8k"))
print(get_dataset_config_names("openai/openai_humaneval"))
print(get_dataset_config_names("google-research-datasets/mbpp"))
ds1 = load_dataset("openai/gsm8k", "main")
print("GSM DATASET SCHEMA")
print("#################")
print(ds1.keys())
print(ds1["test"].features)
for key in ds1["test"].features:
    print(key + ":")
    print(ds1["test"][key][0])

ds2 = load_dataset("openai/openai_humaneval", "openai_humaneval")
print("HUMANEVAL DATASET SCHEMA")
print("#################")
print(ds2.keys())
print(ds2["test"].features)
for key in ds2["test"].features:
    print(key + ":")
    print(ds2["test"][key][0])

ds3 = load_dataset("google-research-datasets/mbpp", "sanitized")
print("MBPP DATASET SCHEMA")
print("#################")
print(ds3.keys())
print(ds3["test"].features)
for key in ds3["test"].features:
    print(key + ":")
    print(ds3["test"][key][0])

print(ds3["test"]["test_imports"][:])

