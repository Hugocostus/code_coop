import streamlit as st
import pandas as pd
from matching.games import HospitalResident
from datetime import datetime
from io import BytesIO


# =============================
# 🔗 LIENS GOOGLE SHEETS
# =============================
URL_COOPTATIONS = "https://docs.google.com/spreadsheets/d/1D1DSeJhV_KOAG7sKblfneWAFyiQbqcDE1FtK4YB_Fmo/export?format=xlsx"
URL_VOEUX_ETUDIANTS = "https://docs.google.com/spreadsheets/d/1oUdR14814s5J_Zhhef00tp_uiLk5fX4ww7AfONVfhG4/export?format=xlsx"
URL_VOEUX_ASSO = "https://docs.google.com/spreadsheets/d/1c7mrDXvjh4QUkaMOi2lX-tXHVgqzFUvd27ku90U1mO8/export?format=xlsx"


# =============================
# 🎯 FONCTION PRINCIPALE
# =============================
def run_matching(sessions_input, date_voeux, heure_voeux):

    # -----------------------------
    # 1️⃣ Charger cooptations
    # -----------------------------
    df = pd.read_excel(URL_COOPTATIONS)
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors='coerce')
    df['Heure'] = pd.to_datetime(df['Heure'], format="%H:%M:%S", errors='coerce').dt.time
    df['DateHeure'] = df.apply(
        lambda r: datetime.combine(r['Date'].date(), r['Heure']) if pd.notna(r['Date']) and pd.notna(r['Heure']) else pd.NaT,
        axis=1
    )
    
    if 'Adresse' in df.columns:
        df = df.drop(columns=['Adresse'])

    sessions = []
    for jour, entree, sortie in sessions_input:
        start = datetime.combine(jour, entree)
        end = datetime.combine(jour, sortie)
        sessions.append((start, end))

    def est_dans_session(dt):
        if pd.isna(dt):
            return False
        return any(start <= dt <= end for start, end in sessions)

    df_filtre = df[df['DateHeure'].apply(est_dans_session)].copy()

    # -----------------------------
    # Cutoff global
    # -----------------------------
    cutoff_voeux = pd.to_datetime(f"{date_voeux} {heure_voeux}")

    # -----------------------------
    # 2️⃣ Voeux étudiants
    # -----------------------------
    df_voeux = pd.read_excel(URL_VOEUX_ETUDIANTS)
    if 'Email' in df_voeux.columns:
        df_voeux = df_voeux.drop(columns=['Email'])

    df_voeux['Datetime'] = pd.to_datetime(
        df_voeux['Date'].astype(str) + ' ' + df_voeux['Heure'].astype(str),
        errors='coerce'
    )

    df_voeux = df_voeux[
        (df_voeux['Datetime'] <= cutoff_voeux) &
        (df_voeux['Etudiant 1'] == df_voeux['Etudiant 2'])
    ]

    df_voeux = df_voeux.sort_values('Datetime', ascending=False).drop_duplicates(subset=['Etudiant 1'], keep='first')
    df_voeux['Numéro étudiant'] = df_voeux['Etudiant 1'].str.extract(r"\(([^)]+)\)")[0].str.strip()
    df_voeux.drop(columns=['Etudiant 1', 'Etudiant 2', 'Date', 'Heure', 'Datetime'], inplace=True, errors='ignore')

    dico_etudiant = {}
    for _, row in df_voeux.iterrows():
        num = row['Numéro étudiant']
        if pd.notna(num) and str(num).isdigit():
            choix = [str(row[col]).strip() for col in df_voeux.columns if col != 'Numéro étudiant' and pd.notna(row[col])]
            choix = list(dict.fromkeys(choix))
            dico_etudiant[int(num)] = choix

    # Couperet des sessions : on filtre strictement les étudiants validés
    numeros_autorises = set(df_filtre['Numero'].dropna().astype(int))
    dico_etudiant = {etu: prefs for etu, prefs in dico_etudiant.items() if etu in numeros_autorises}

    # -----------------------------
    # 3️⃣ Voeux associations
    # -----------------------------
    asso = pd.read_excel(URL_VOEUX_ASSO)
    if 'Email' in asso.columns:
        asso = asso.drop(columns=['Email'])

    asso['datetime'] = pd.to_datetime(
        asso['Date'].astype(str) + ' ' + asso['Heure'].astype(str),
        errors='coerce'
    )
    asso = asso[asso['datetime'] <= cutoff_voeux]
    asso = asso.sort_values('datetime').drop_duplicates(subset='Association', keep='last')

    asso['Association'] = asso['Association'].astype(str).str.strip()
    
    # Capacité en entiers stricts
    nb_liste_asso = {
        row['Association']: int(row['Numero'])
        for _, row in asso.iterrows()
        if pd.notna(row['Numero']) and int(row['Numero']) > 0
    }

    asso['Etudiant_split'] = asso['Etudiant'].astype(str).str.split(',')
    asso['Etudiant_split'] = asso['Etudiant_split'].apply(
        lambda x: x + [None] * (150 - len(x)) if len(x) < 150 else x[:150]
    )

    etudiant_cols = [f"etudiant {i}" for i in range(1, 151)]
    etudiants_expanded = pd.DataFrame(asso['Etudiant_split'].tolist(), columns=etudiant_cols)

    voeux_asso_finaux = pd.concat([asso[['Association']].reset_index(drop=True), etudiants_expanded], axis=1)
    voeux_asso_finaux.rename(columns={'Association': 'asso'}, inplace=True)

    # -----------------------------
    # 4️⃣ Normalisation des numéros
    # -----------------------------
    etudiants = df_filtre.copy()
    etudiants['nom_prenom'] = etudiants['Prenom'].astype(str).str.strip().str.lower() + ' ' + etudiants['Nom'].astype(str).str.strip().str.lower()
    etudiants['prenom_nom'] = etudiants['Nom'].astype(str).str.strip().str.lower() + ' ' + etudiants['Prenom'].astype(str).str.strip().str.lower()
    
    mapping_nom = {}
    for _, r in etudiants.iterrows():
        if pd.notna(r['Numero']):
            mapping_nom[r['nom_prenom']] = int(r['Numero'])
            mapping_nom[r['prenom_nom']] = int(r['Numero'])

    def normaliser_num(val):
        if pd.isna(val):
            return None
        val_str = str(val).strip()
        match = pd.Series([val_str]).str.extract(r'(\d+)')[0].dropna()
        if not match.empty:
            return int(match.iloc[0])
        return mapping_nom.get(val_str.lower(), None)

    asso_to_numeros = {}
    for _, row in voeux_asso_finaux.iterrows():
        asso_name = row['asso']
        numeros = [normaliser_num(val) for val in row[1:] if pd.notna(val)]
        numeros_valides = [int(n) for n in numeros if n is not None]
        asso_to_numeros[asso_name] = list(dict.fromkeys(numeros_valides))

    # -----------------------------
    # 5️⃣ Nettoyage mutuel récursif
    # -----------------------------
    while True:
        len_etu = len(dico_etudiant)
        len_asso = len(asso_to_numeros)

        assos_possibles = set(asso_to_numeros.keys()).intersection(nb_liste_asso.keys())

        # Nettoyage choix étudiants
        nouveau_dico_etudiant = {}
        for etu, prefs in dico_etudiant.items():
            choix = [a for a in prefs if a in assos_possibles and etu in asso_to_numeros.get(a, [])]
            if choix:
                nouveau_dico_etudiant[etu] = choix
        dico_etudiant = nouveau_dico_etudiant

        # Nettoyage choix assos
        etus_possibles = set(dico_etudiant.keys())
        nouveau_asso_to_numeros = {}
        for a, nums in asso_to_numeros.items():
            if a in assos_possibles:
                cands = [n for n in nums if n in etus_possibles and a in dico_etudiant.get(n, [])]
                if cands:
                    nouveau_asso_to_numeros[a] = cands
        asso_to_numeros = nouveau_asso_to_numeros

        nb_liste_asso = {a: cap for a, cap in nb_liste_asso.items() if a in asso_to_numeros}

        if len(dico_etudiant) == len_etu and len(asso_to_numeros) == len_asso:
            break

    # -----------------------------
    # 6️⃣ Matching
    # -----------------------------
    if not dico_etudiant or not asso_to_numeros:
        st.error("Aucune correspondance réciproque valide trouvée pour effectuer le matching.")
        st.stop()

    game = HospitalResident.create_from_dictionaries(dico_etudiant, asso_to_numeros, nb_liste_asso)
    cooptes = game.solve()

    rows = []
    for asso_obj, etudiants_objs in cooptes.items():
        nom_asso = getattr(asso_obj, 'name', str(asso_obj))
        candidats = [str(getattr(e, 'name', e)) for e in etudiants_objs]
        rows.append([nom_asso] + candidats)

    max_len = max((len(r) for r in rows), default=1)
    
    if max_len > 1:
        rows_padded = [r + [""] * (max_len - len(r)) for r in rows]
        colonnes = ['Asso'] + [f'Coopté_{i}' for i in range(1, max_len)]
        df_cooptes = pd.DataFrame(rows_padded, columns=colonnes)
    else:
        df_cooptes = pd.DataFrame([[r[0]] for r in rows], columns=['Asso'])

    # -----------------------------
    # 7️⃣ Remplacement numéros -> prénoms
    # -----------------------------
    df_voeux_noms = pd.read_excel(URL_VOEUX_ETUDIANTS)
    df_voeux_noms["Prenom"] = df_voeux_noms["Etudiant 1"].str.extract(r"^([^()]+)\(")[0].str.strip()
    df_voeux_noms["Numero"] = df_voeux_noms["Etudiant 1"].str.extract(r"\(([^)]+)\)")[0].str.strip()
    
    mapping_numero_prenom = dict(zip(df_voeux_noms["Numero"].dropna(), df_voeux_noms["Prenom"].dropna()))

    for col in df_cooptes.columns[1:]:
        df_cooptes[col] = df_cooptes[col].map(mapping_numero_prenom).fillna(df_cooptes[col])

    # -----------------------------
    # 8️⃣ Export en mémoire
    # -----------------------------
    buffer_cooptes = BytesIO()
    df_cooptes.to_excel(buffer_cooptes, index=False)
    buffer_cooptes.seek(0)

    buffer_dicos = BytesIO()
    with pd.ExcelWriter(buffer_dicos) as writer:
        pd.DataFrame([(k, v) for k, v in asso_to_numeros.items()], columns=["asso", "numeros"]).to_excel(writer, sheet_name="asso_to_numeros", index=False)
        pd.DataFrame(list(nb_liste_asso.items()), columns=["asso", "capacite"]).to_excel(writer, sheet_name="nb_liste_asso", index=False)
        pd.DataFrame([(k, v) for k, v in dico_etudiant.items()], columns=["num_etudiant", "choix"]).to_excel(writer, sheet_name="dico_etudiant", index=False)
    buffer_dicos.seek(0)

    return buffer_cooptes, buffer_dicos


# =============================
# 🌐 INTERFACE STREAMLIT
# =============================
st.set_page_config(page_title="Matching Cooptations", layout="wide")
st.title("Plateforme Matching Cooptations")

st.header("Sessions de cooptation")
nb_sessions = st.number_input("Nombre de sessions", min_value=1, step=1, value=1)

sessions_input = []
for i in range(nb_sessions):
    st.subheader(f"Session {i+1}")
    col1, col2, col3 = st.columns(3)
    with col1:
        jour = st.date_input("Jour", key=f"date{i}")
    with col2:
        entree = st.time_input("Heure entrée", key=f"entree{i}")
    with col3:
        sortie = st.time_input("Heure sortie", key=f"sortie{i}")
    sessions_input.append((jour, entree, sortie))

st.header("Date limite des vœux")
col1, col2 = st.columns(2)
with col1:
    date_voeux = st.date_input("Date limite")
with col2:
    heure_voeux = st.time_input("Heure limite")

if st.button("Lancer le matching", type="primary"):
    with st.spinner("Matching en cours..."):
        try:
            cooptes_file, dicos_file = run_matching(
                sessions_input,
                date_voeux,
                heure_voeux
            )
            st.success("Matching terminé avec succès !")
            
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label="📥 Télécharger export_cooptes.xlsx",
                    data=cooptes_file,
                    file_name="export_cooptes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with c2:
                st.download_button(
                    label="📥 Télécharger export_dicos.xlsx",
                    data=dicos_file,
                    file_name="export_dicos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'exécution : {e}")



