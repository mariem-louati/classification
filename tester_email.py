import joblib

modele = joblib.load("modele_emails.pkl")

print("Écris un e-mail (ou tape q pour quitter)")

while True:
    texte = input("\nE-mail : ")
    if texte.lower() == "q":
        break
    categorie = modele.predict([texte])[0]
    confiance = max(modele.predict_proba([texte])[0])
    print("Catégorie :", categorie, "(confiance :", round(confiance * 100), "%)")