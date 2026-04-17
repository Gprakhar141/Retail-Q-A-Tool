import ast
import logging
import os

# Keep terminal output clean from known third-party advisory/watcher noise.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
from langchain_helper import get_few_shot_db_chain

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.title("T Shirts Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    chain = get_few_shot_db_chain()
    response = chain.invoke(question)
    cleaned = response['result'].replace("Decimal('", "").replace("')", "")
    value = ast.literal_eval(cleaned)[0][0]

    st.header("Answer")
    st.subheader(int(value))