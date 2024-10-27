import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import openai

# from llama_index.core import PromptTemplate

openai.api_key = st.secrets.openai_key
dimensionen = (
    "Erfassen von Lerninhalten",
    "Arbeitsweise und Arbeitsverhalten",
    "Sozialverhalten",
)
text_qa_template_str = "Kontext information"


@st.cache_resource(show_spinner=True)
def load_data():
    with st.spinner(text="Bewertungsformulierungen werden eingelesen."):
        reader = SimpleDirectoryReader(input_dir="zeugnisbot/data", recursive=True)
        docs = reader.load_data()
        Settings.llm = OpenAI(
            model="gpt-4o",
            temperature=0.4,
            system_prompt=f"Du bist ein Lehrer und schreibst verbale Einschätzungen für Schüler. Folgende Kategorien werden bewertet: {str(dimensionen)}. In der Bewertungstabelle sind den Noten 1 bis 4 Formulierungen für die Kategorien zugeordnet. Gib für jede Kategorie eine verbale Einschätzung die auf den Formulierungen aus der passenden Kategorie und Note basiert. Verwende die korrekten geschlechtlichen Pronomen er/ihn/sein für Jungen, sie/ihr/ihr für Mädchen. Verwende den Name des Schülers oder der Schülerin und ergänze die individuelle Bemerkung am Ende.",
        )

        index = VectorStoreIndex.from_documents(docs)
        return index


index = load_data()
query_engine = index.as_query_engine()

st.title("Zeugnisbot")


bewertungen = {}

with st.form("zeugnisformular"):

    name = st.text_input("Name", placeholder="Name des Schülers/der Schülerin")
    geschlecht = st.radio("Geschlecht", ["Junge", "Mädchen"])

    for dimension in dimensionen:
        bewertungen[dimension] = st.select_slider(
            dimension, options=list(range(1, 5)), value=1
        )

    kommentar = st.text_input("Bemerkung", placeholder="individuelle Bemerkung")
    submitted = st.form_submit_button("Zeugnis erstellen")
    if submitted:
        response = query_engine.query(
            f"Schreibe eine Bewertung für folgenden Schüler: Name: {name}, Geschlecht: {geschlecht}, individuelle Bemerkung: {kommentar}, {', '.join(f'{key}: {value}' for key, value in bewertungen.items())}"
        )

        st.markdown(response.response)
