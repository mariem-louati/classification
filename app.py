import streamlit as st
import joblib

modele = joblib.load("modele_emails.pkl")

st.title("Classification automatique des e-mails")
texte = st.text_area("Colle un e-mail ici :", height=200)

if st.button("Classer"):
    if texte.strip() == "":
        st.warning("Écris un e-mail d'abord.")
    else:
        categorie = modele.predict([texte])[0]
        confiance = max(modele.predict_proba([texte])[0])
        st.success(f"Catégorie : {categorie}")
        st.write(f"Confiance : {round(confiance * 100)} %")