# src/merge_raw.py
import pandas as pd
import os

raw_dir = "data/raw"
fake = pd.read_csv(os.path.join(raw_dir, "Fake.csv"))
true = pd.read_csv(os.path.join(raw_dir, "True.csv"))

fake['label'] = 0   # fake -> 0
true['label'] = 1   # real -> 1

# Some Kaggle files might have columns 'title','text' or 'subject' - keep both
fake = fake.rename(columns=lambda x: x.strip())
true = true.rename(columns=lambda x: x.strip())

# Keep columns 'title' and 'text' if present else combine 'title'+'text'
fake['text'] = fake['title'].fillna('') + ". " + fake['text'].fillna('')
true['text'] = true['title'].fillna('') + ". " + true['text'].fillna('')

df = pd.concat([true[['text','label']], fake[['text','label']]], ignore_index=True)
df.to_csv("data/processed/combined.csv", index=False)
print("Saved combined dataset:", len(df))
print(df['label'].value_counts())
