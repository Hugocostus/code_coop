import streamlit as st
import pandas as pd
from matching.games import HospitalResident
from datetime import datetime
from io import BytesIO


# =============================
# 🔗 LIENS GOOGLE SHEETS
# =============================
URL_COOPTATIONS = "https://docs.google.com/spreadsheets/d/1rxGm0HY-8hghBFIPiZf0TNlk9XcxwBraoj77-kOLXjI/export?format=xlsx"
URL_VOEUX_ETUDIANTS = "https://docs.google.com/spreadsheets/d/1hxTFNoBHznWh408UwHM6dy6csJbvcEv-q4odBIy-5Ck/export?format=xlsx"
URL_VOEUX_ASSO = "https://docs.google.com/spreadsheets/d/1bO6xNI1wOfupyzbK3zBZYLbSeTcXgPpeT0puLArzCZs/export?format=xlsx"

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
    df['DateHeure'] = df.apply(lambda r: datetime.combine(r['Date'].date(), r['Heure']), axis=1)
    df = df.drop(columns=['Adresse'])

    sessions = []
    for jour, entree, sortie in sessions_input:
        start = datetime.combine(jour, entree)
        end = datetime.combine(jour, sortie)
        sessions.append((start, end))

    def est_dans_session(dt):
        return any(start <= dt <= end for start, end in sessions)

    df_filtre = df[df['DateHeure'].apply(est_dans_session)]

    # -----------------------------
    # Cutoff global
    # -----------------------------
    cutoff_voeux = pd.to_datetime(f"{date_voeux} {heure_voeux}")

    # -----------------------------
    # 2️⃣ Voeux étudiants
    # -----------------------------
    df_voeux = pd.read_excel(URL_VOEUX_ETUDIANTS)
    df_voeux = df_voeux.drop(columns=['Email'])
    df_voeux['Datetime'] = pd.to_datetime(
        df_voeux['Date'].astype(str) + ' ' + df_voeux['Heure'].astype(str),
        errors='coerce'
    )

    df_voeux = df_voeux[
        (df_voeux['Datetime'] <= cutoff_voeux) &
        (df_voeux['Etudiant 1'] == df_voeux['Etudiant 2'])
    ]

    df_voeux = df_voeux.sort_values('Datetime', ascending=False) \
                       .drop_duplicates(subset=['Etudiant 1'], keep='first')

    df_voeux['Numéro étudiant'] = df_voeux['Etudiant 1'].str.extract(r"\(([^)]+)\)")[0].str.strip()
    df_voeux.drop(columns=['Etudiant 1','Etudiant 2','Date','Heure','Datetime'], inplace=True)

    dico_etudiant = {}
    for _, row in df_voeux.iterrows():
        num = row['Numéro étudiant']
        choix = [row[col] for col in df_voeux.columns if col != 'Numéro étudiant' and pd.notna(row[col])]
        if num.isdigit():
            dico_etudiant[int(num)] = choix

    # -----------------------------
    # 3️⃣ Voeux associations
    # -----------------------------
    asso = pd.read_excel(URL_VOEUX_ASSO)
    asso['Etudiant_split'] = asso['Etudiant'].str.split(',')
    asso = asso.drop(columns=['Etudiant']).drop_duplicates(subset='Association', keep='last')
    asso = asso.drop(columns=['Email'])
    asso['datetime'] = pd.to_datetime(
        asso['Date'].astype(str) + ' ' + asso['Heure'].astype(str),
        errors='coerce'
    )
    asso = asso[asso['datetime'] <= cutoff_voeux]
    nb_liste_asso = dict(zip(asso['Association'], asso['Numero']))
    asso['Etudiant_split'] = asso['Etudiant_split'].apply(lambda x: x + [None] * (150 - len(x)) if len(x) < 150 else x)

    etudiant_cols = [f"etudiant {i}" for i in range(1,151)]
    etudiants_expanded = pd.DataFrame(asso['Etudiant_split'].tolist(), columns=etudiant_cols)

    voeux_asso_finaux = pd.concat([asso[['Association']].reset_index(drop=True), etudiants_expanded], axis=1)
    voeux_asso_finaux.rename(columns={'Association':'asso'}, inplace=True)

    # -----------------------------
    # 4️⃣ Normalisation
    # -----------------------------
    etudiants = df_filtre.copy()
    etudiants['nom_prenom'] = etudiants['Prenom'].astype(str).str.strip() + ' ' + etudiants['Nom'].astype(str).str.strip()
    etudiants['all'] = etudiants['Prenom'].astype(str) + ' ' + etudiants['Nom'].astype(str) + ' ' + etudiants['Numero'].astype(str)
    nom_prenom_to_all = etudiants.set_index('nom_prenom')['all'].to_dict()

    for col in voeux_asso_finaux.columns:
        if col != 'asso':
            voeux_asso_finaux[col] = voeux_asso_finaux[col].astype(str).str.strip()
            voeux_asso_finaux[col] = voeux_asso_finaux[col].map(nom_prenom_to_all).fillna(voeux_asso_finaux[col])
            voeux_asso_finaux[col] = voeux_asso_finaux[col].astype(str).str.extract(r'(\d+)').astype('Int64')

    asso_to_numeros = {}
    for _, row in voeux_asso_finaux.iterrows():
        asso_name = row['asso']
        numeros = [int(num) for num in row[1:] if pd.notna(num)]
        asso_to_numeros[asso_name] = numeros

    # -----------------------------
    # Nettoyage mutuel
    # -----------------------------
    for etu in list(dico_etudiant.keys()):
        choix_valides = [asso_name for asso_name in dico_etudiant[etu] if etu in asso_to_numeros.get(asso_name, [])]
        if choix_valides:
            dico_etudiant[etu] = choix_valides
        else:
            del dico_etudiant[etu]

    for asso_name in list(asso_to_numeros.keys()):
        etudiants_valides = [etu for etu in asso_to_numeros[asso_name] if etu in dico_etudiant]
        if etudiants_valides:
            asso_to_numeros[asso_name] = etudiants_valides
        else:
            del asso_to_numeros[asso_name]

    # -----------------------------
    # Matching
    # -----------------------------
    game = HospitalResident.create_from_dictionaries(dico_etudiant, asso_to_numeros, nb_liste_asso)
    cooptes = game.solve()

    rows = []
    for etu, assos in cooptes.items():
        rows.append([etu] + assos)

    max_len = max(len(r) for r in rows)
    rows = [r + [None]*(max_len - len(r)) for r in rows]

    df_cooptes = pd.DataFrame(rows)
    df_cooptes.columns = ['Asso'] + [f'Coopté_{i}' for i in range(1, max_len)]

    # -----------------------------
    # Remplacement numéros -> prénoms
    # -----------------------------
    df_voeux_noms = pd.read_excel(URL_VOEUX_ETUDIANTS)
    df_voeux_noms["Prenom"] = df_voeux_noms["Etudiant 1"].str.extract(r"^([^()]+)\(")[0].str.strip()
    df_voeux_noms["Numero"] = df_voeux_noms["Etudiant 1"].str.extract(r"\(([^)]+)\)")[0]
    mapping_numero_prenom = dict(zip(df_voeux_noms["Numero"], df_voeux_noms["Prenom"]))

    for col in df_cooptes.columns[1:]:
        df_cooptes[col] = df_cooptes[col].astype(str)
        df_cooptes[col] = df_cooptes[col].map(mapping_numero_prenom).fillna(df_cooptes[col])
        df_cooptes = df_cooptes.replace("None", "")

    # -----------------------------
    # Export en mémoire
    # -----------------------------
    buffer_cooptes = BytesIO()
    df_cooptes.to_excel(buffer_cooptes, index=False)
    buffer_cooptes.seek(0)

    buffer_dicos = BytesIO()
    with pd.ExcelWriter(buffer_dicos) as writer:
        pd.DataFrame([(k,v) for k,v in asso_to_numeros.items()],
                     columns=["asso","numeros"]).to_excel(writer, sheet_name="asso_to_numeros", index=False)
        pd.DataFrame(list(nb_liste_asso.items()),
                     columns=["asso","capacite"]).to_excel(writer, sheet_name="nb_liste_asso", index=False)
        pd.DataFrame([(k,v) for k,v in dico_etudiant.items()],
                     columns=["num_etudiant","choix"]).to_excel(writer, sheet_name="dico_etudiant", index=False)
    buffer_dicos.seek(0)

    return buffer_cooptes, buffer_dicos

# =============================
# 🌐 INTERFACE
# =============================
st.title("Plateforme Matching Cooptations")

st.header("Sessions de cooptation")
nb_sessions = st.number_input("Nombre de sessions", min_value=1, step=1)

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

if st.button("Lancer le matching"):
    with st.spinner("Matching en cours..."):
        cooptes_file, dicos_file = run_matching(
            sessions_input,
            date_voeux,
            heure_voeux
        )
    st.success("Matching terminé !")
    st.download_button("Télécharger export_cooptes.xlsx", cooptes_file, file_name="export_cooptes.xlsx")
    st.download_button("Télécharger export_dicos.xlsx", dicos_file, file_name="export_dicos.xlsx")





