import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/emails_5_classes.csv")
train = df[df["split"] == "train"]

modele = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    LogisticRegression(max_iter=1000),
)

modele.fit(train["email"], train["categorie"])

joblib.dump(modele, "modele_emails.pkl")
print("Modèle entraîné avec", len(train), "e-mails et sauvegardé.")