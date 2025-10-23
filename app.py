import os
import json
import re
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

# --------- config ----------
load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# clients
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY),
)

oai = OpenAI(api_key=OPENAI_API_KEY)

# --------- helpers ----------
NAV_GARBAGE = [
    "Skip to main content", "Back to top", "User Guide", "API reference",
    "Development", "Release notes", "Learn", "NEPs", "Choose version",
    "GitHub", "Built with the PyData Sphinx Theme", "Created using Sphinx",
    "On this page", "Section Navigation", "Go Back", "Open In Tab",
]

def maybe_deserialize_chunk(text: str) -> str:
    """
    Your 'chunk' field currently sometimes comes back as a JSON array in a string, e.g. "[\"...\",\"...\"]".
    If it starts with '[' and parses as JSON -> join lines. Otherwise return raw.
    """
    t = (text or "").strip()
    if t.startswith("[") and t.endswith("]"):
        try:
            arr = json.loads(t)
            if isinstance(arr, list):
                # join with double newlines but clip super long nav sections
                return "\n\n".join(str(x) for x in arr)
        except Exception:
            pass
    return t

def light_clean(text: str) -> str:
    # remove repeated whitespace + common nav garbage
    t = re.sub(r"\s+\n", "\n", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    for g in NAV_GARBAGE:
        t = t.replace(g, "")
    return t.strip()

def search_docs(query: str, k: int = 6) -> List[Dict]:
    """
    Simple keyword search 
    """
    results = search_client.search(
        search_text=query if query.strip() else "*",
        top=k,
        include_total_count=False,
    )
    docs = []
    for r in results:
        chunk = r.get("chunk", "")
        chunk = maybe_deserialize_chunk(chunk)
        chunk = light_clean(chunk)
        url = r.get("url") or ""
        docs.append({"chunk": chunk, "url": url})
    # filter empty chunks
    return [d for d in docs if d["chunk"]]

def build_prompt(query: str, contexts: List[Dict]) -> str:
    joined = "\n\n---\n\n".join(f"Source: {c['url']}\n{c['chunk'][:3000]}" for c in contexts)
    return (
        "You are a careful Python documentation assistant. "
        "Answer using only the information in the provided sources. "
        "If the answer isn't in the sources, say you don't know.\n\n"
        f"Question: {query}\n\n"
        f"Sources:\n{joined}\n\n"
        "Now provide a concise, correct answer. Include short citations like [1], [2] using the source order below."
    )

def ask_llm(query: str, contexts: List[Dict]) -> str:
    prompt = build_prompt(query, contexts)
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def format_citations(answer: str, contexts: List[Dict]) -> str:
    # simple 1..N mapping
    numbered = "\n".join(f"[{i+1}] {c['url']}" for i, c in enumerate(contexts))
    return f"{answer}\n\n**Sources**\n{numbered}"

# --------- UI ----------
st.set_page_config(page_title="Python Docs RAG", page_icon="🐍", layout="wide")
st.title("🐍 Python Docs RAG (Azure Search + OpenAI)")

with st.sidebar:
    st.caption("Search settings")
    topk = st.slider("Top chunks", 2, 12, 6, 1)
    st.caption("Tip: once you add embeddings + semantic, you can switch to vector/hybrid search.")

tab1, tab2 = st.tabs(["Ask", "Raw search"])

with tab2:
    q = st.text_input("Raw search query (hits from index):", "read_csv")
    if st.button("Search", key="raw"):
        hits = search_docs(q, k=topk)
        if not hits:
            st.warning("No chunks returned. Try another query.")
        for i, h in enumerate(hits, start=1):
            with st.expander(f"Hit {i} — {h['url']}"):
                st.write(h["chunk"][:2000])

with tab1:
    query = st.text_input("Ask a question about the docs:")
    if st.button("Answer", type="primary"):
        ctx = search_docs(query, k=topk)
        if not ctx:
            st.error("No context chunks found. Check your index data.")
        else:
            answer = ask_llm(query, ctx)
            st.markdown(format_citations(answer, ctx))
