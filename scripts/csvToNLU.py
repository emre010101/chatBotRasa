import pandas as pd

# Load dataset
dataset_path = "data/Synthetic_Intent___NER_Dataset (last).csv"  # Update your dataset path
df = pd.read_csv(dataset_path)

# Group queries by intent
nlu_data = {}
for _, row in df.iterrows():
    intent = row["intent"]
    query = row["query"]

    if intent not in nlu_data:
        nlu_data[intent] = []

    nlu_data[intent].append(f"- {query}")  # Format query for Rasa

# Write to `nlu.yml`
with open("rasa_model/nlu2.yml", "w", encoding="utf-8") as f:
    f.write("version: '3.1'\n\nnlu:\n")

    for intent, queries in nlu_data.items():
        f.write(f"- intent: {intent}\n  examples: |\n")
        for query in queries:
            f.write(f"    {query}\n")

print("✅ Rasa NLU dataset saved as `nlu.yml`")
