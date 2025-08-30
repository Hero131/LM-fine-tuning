import streamlit as st
import os
import tempfile
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Clash of Contexts - Ingestion & Query", layout="wide")
st.title("Clash of Contexts: Magical Ingestion Panel")

# -------------------------
# Session State
# -------------------------
if "magic_scrolls" not in st.session_state:
    st.session_state.magic_scrolls = []
if "battle_log" not in st.session_state:
    st.session_state.battle_log = []

SPELLBOOK_DIRECTORY = "./arcane_chroma_vault"
os.makedirs(SPELLBOOK_DIRECTORY, exist_ok=True)

# -------------------------
# Initialize Magical Embeddings & Vault
# -------------------------
@st.cache_resource
def summon_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

magical_thingy = summon_embeddings()

@st.cache_resource
def summon_vault():
    return Chroma(persist_directory=SPELLBOOK_DIRECTORY, embedding_function=magical_thingy)

freezing_spell = summon_vault()

# -------------------------
# Initialize Wizard (LLM)
# -------------------------
@st.cache_resource
def summon_wizard():
    return Ollama(model="llama2")  #  Default wizard for answering riddles
    # For customization, uncomment:
    # wizard_choice = st.sidebar.selectbox("Choose Your Wizard", ["llama2", "mistral", "phi"])
    # return Ollama(model=wizard_choice)

wise_wizard = summon_wizard()

# -------------------------
# Spellcasting: Ingest Scrolls
# -------------------------
def cast_ingestion_spell(scroll, rune_size, rune_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{scroll.name.split('.')[-1]}") as temp_scroll:
        temp_scroll.write(scroll.getvalue())
        scroll_path = temp_scroll.name
    try:
        if scroll.name.endswith('.pdf'):
            summoner = PyPDFLoader(scroll_path)
        elif scroll.name.endswith('.docx'):
            summoner = Docx2txtLoader(scroll_path)
        elif scroll.name.endswith('.txt'):
            summoner = TextLoader(scroll_path)
        elif scroll.name.endswith('.md'):
            summoner = UnstructuredMarkdownLoader(scroll_path)
        else:
            os.unlink(scroll_path)
            return None, f" Forbidden artifact: {scroll.name}"

        parchments = summoner.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=rune_size,
            chunk_overlap=rune_overlap
        )
        fragments = splitter.split_documents(parchments)

        for i, frag in enumerate(fragments):
            frag.metadata["source"] = scroll.name
            frag.metadata["fragment_id"] = i

        freezing_spell.add_documents(fragments)
        freezing_spell.persist()
        os.unlink(scroll_path)
        return fragments, None

    except Exception as e:
        os.unlink(scroll_path)
        return None, str(e)

# -------------------------
# Sidebar Parameters
# -------------------------
st.sidebar.header(" Spell Parameters")
rune_size = st.sidebar.slider("Rune Size", min_value=100, max_value=2000, value=500, step=100)
rune_overlap = st.sidebar.slider("Rune Overlap", min_value=0, max_value=500, value=50, step=10)

# Clear vault option
st.sidebar.markdown("---")
if st.sidebar.button(" Purge the Vault"):
    if st.sidebar.checkbox("I swear upon my clan to erase all scrolls"):
        try:
            freezing_spell = Chroma(persist_directory=SPELLBOOK_DIRECTORY, embedding_function=magical_thingy)
            freezing_spell.delete_collection()
            freezing_spell = Chroma(persist_directory=SPELLBOOK_DIRECTORY, embedding_function=magical_thingy)
            st.session_state.magic_scrolls = []
            st.session_state.battle_log = []
            st.success(" The Vault has been purged!")
        except Exception as e:
            st.sidebar.error(f"Dark forces blocked the purge: {str(e)}")

# -------------------------
# Tabs for workflow
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs([" Upload Scrolls", " Query Vault", " Wizard Chat", " Vault Info"])

# --- Upload Tab ---
with tab1:
    st.subheader("Upload Your Magical Scrolls")
    uploaded_scrolls = st.file_uploader("Choose your scrolls", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)

    if uploaded_scrolls and st.button("✨ Cast Ingestion Spell"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_scrolls = len(uploaded_scrolls)
        successful_scrolls = 0

        for i, scroll in enumerate(uploaded_scrolls):
            status_text.text(f"Enchanting {scroll.name}...")
            fragments, curse = cast_ingestion_spell(scroll, rune_size, rune_overlap)

            if curse:
                st.session_state.magic_scrolls.append({"type": "error", "content": f"{curse}"})
            else:
                successful_scrolls += 1
                st.session_state.magic_scrolls.append({"type": "success", "content": f" {scroll.name}: {len(fragments)} fragments crafted"})

            progress_bar.progress((i + 1) / total_scrolls)

        status_text.text(f"Ritual complete! {successful_scrolls}/{total_scrolls} scrolls enchanted.")

    for msg in st.session_state.magic_scrolls:
        if msg["type"] == "error":
            st.error(msg["content"])
        else:
            st.success(msg["content"])

# --- Query Tab ---
with tab2:
    st.subheader("Query the Arcane Vault")
    query_spell = st.text_input("Whisper your query:")

    if query_spell and st.button("Scry the Vault"):
        with st.spinner("Casting divination..."):
            results = freezing_spell.similarity_search(query_spell, k=5)
            st.write(f"### Top {len(results)} Findings")
            for i, frag in enumerate(results):
                with st.expander(f"Finding {i+1} - From {frag.metadata.get('source', 'Unknown Scroll')}"):
                    st.write(frag.page_content)
                    st.caption(f"Metadata: {frag.metadata}")

# --- Chat Mode Tab ---
with tab3:
    st.subheader("Converse with the Wizard Guardian")
    incantation = st.chat_input("Ask your riddle to the wizard...")

    if incantation:
        st.session_state.battle_log.append({"role": "hero", "content": incantation})

        # Retrieve fragments
        fragments = freezing_spell.similarity_search(incantation, k=3)
        context = "\n\n".join([frag.page_content for frag in fragments]) if fragments else ""

        # Wizard answers using LLM
        spell_prompt = f"You are a wise wizard with access to ancient scrolls.\n\nContext:\n{context}\n\nRiddle: {incantation}\n\nAnswer with wisdom:"
        prophecy = wise_wizard.invoke(spell_prompt)

        st.session_state.battle_log.append({"role": "wizard", "content": prophecy})

    for chant in st.session_state.battle_log:
        if chant["role"] == "hero":
            with st.chat_message("user"):
                st.write(chant["content"])
        else:
            with st.chat_message("assistant"):
                st.write(chant["content"])

# --- Database Info Tab ---
with tab4:
    st.subheader("Vault of Knowledge Stats")
    try:
        collection = freezing_spell._collection
        count = collection.count()
        st.metric("Scroll Fragments Stored", count)
    except:
        st.warning("Could not peer into the vault.")
