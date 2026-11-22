import pandas as pd

df = pd.read_csv("data/emails_5_classes.csv")

print("Taille :", df.shape)
print("Colonnes :", df.columns.tolist())
print(df["categorie"].value_counts())
print("Doublons :", df.duplicated(subset=["email"]).sum())
print("Vides :", df["email"].isna().sum())
print(df["split"].value_counts())