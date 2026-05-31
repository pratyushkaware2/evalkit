from evalkit.adapters import GSM8KAdapter, MBPPAdapter, HumanEvalAdapter

print(GSM8KAdapter().normalized("test"))
print(HumanEvalAdapter().normalized("test"))
print(MBPPAdapter().normalized("test"))
