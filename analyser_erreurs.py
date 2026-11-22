import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
modele = joblib.load("modele_emails.pkl")
df = pd.read_csv("data/emails_5_classes.csv")

for nom in ["validation", "test", "test_manuel"]:
    d = df[df["split"] == nom]
    prediction = modele.predict(d["email"])
    score = accuracy_score(d["categorie"], prediction)
    print(nom, ": bonnes réponses =", round(score * 100, 1), "%")

d = df[df["split"] == "test_manuel"].copy()
d["prediction"] = modele.predict(d["email"])

print("\nTableau des erreurs (lignes = vraie catégorie, colonnes = ce que le modèle a répondu) :")
print(pd.crosstab(d["categorie"], d["prediction"]))

print("\nE-mails où le modèle se trompe :")
erreurs = d[d["categorie"] != d["prediction"]]
for _, ligne in erreurs.iterrows():
    print("-", ligne["email"])
    print("   vrai :", ligne["categorie"], "| répondu :", ligne["prediction"])
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    d["categorie"], d["prediction"], xticks_rotation=30, cmap="Blues"
)
plt.title("Résultats sur les 60 e-mails écrits à la main")
plt.xlabel("Réponse du modèle")
plt.ylabel("Vraie catégorie")
plt.tight_layout()
plt.savefig("resultats_test_manuel.png")
plt.show()