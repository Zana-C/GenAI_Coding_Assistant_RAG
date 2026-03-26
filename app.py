# app.py - ÇOK DİLLİ VE ÇOK KATMANLI KODLAMA ASİSTANI

import re
import streamlit as st
from rag_pipeline import initialize_rag_system
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA

# --- A. SİSTEM BAŞLATMA (CACHE) ---
@st.cache_resource
def setup_system():
    try:
        # rag_pipeline'dan vektör depolarını ve LLM'i yükler
        _, vector_stores, llm = initialize_rag_system()
        return vector_stores, llm
    except Exception as e:
        st.error(f"RAG Sistemi Başlatılamadı! Detay: {e}")
        st.stop()

vector_stores, llm = setup_system()

# --- B. PROMPT ŞABLONLARI ---

# 1. Yönlendirici (Router) Prompt'u
CLASSIFIER_PROMPT = """Analyze the user query and classify it into one of the following categories:
- 'kodlama_rehberi': If the query is about writing code, syntax, SQL, or programming help.
- 'hata_ayiklama_asistani': If the query is about specific errors, bugs, or troubleshooting.

Output ONLY the category name.

Query: {query}
Category:"""

# 2. Mentor (QA) Prompt'u (Hatanın çözüldüğü yer: {query} -> {question})
QA_PROMPT_TEMPLATE = """You are a highly skilled software mentor. 
Answer the user's question using ONLY the provided context.

1. Respond in the SAME language as the user's question (e.g., if Turkish -> answer Turkish, if Japanese -> answer Japanese).
2. If the context is in English and the question is in another language, translate and explain technically.
3. Provide code examples in clear code blocks.
4. If the answer is not in the context, say: "I don't have information on this in my database."

Context: {context}
Question: {question} 
Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE, 
    input_variables=["context", "question"]
)

# --- C. ZİNCİRLER VE MANTIK ---

classifier_chain = (
    PromptTemplate.from_template(CLASSIFIER_PROMPT)
    | llm
    | StrOutputParser()
)

def get_response(query: str):
    try:
        # 1. Adım: Sorgu sınıflandırma
        raw_topic = classifier_chain.invoke({"query": query})

        # LLM bazen backtick, tırnak veya fazladan metin ekleyebilir.
        # Bilinen kategori adlarından birini bulmak için regex kullanıyoruz.
        topic = None
        raw_lower = raw_topic.strip().lower()
        for key in vector_stores.keys():
            # Anahtar kelimeyi düz veya işaretlenmiş hâliyle ara
            if re.search(r'\b' + re.escape(key) + r'\b', raw_lower):
                topic = key
                break

        # Eğer hiçbir kategori eşleşmezse, ilk mevcut depoya yönlendir
        if topic is None:
            topic = list(vector_stores.keys())[0]
            st.warning(f"Kategori belirlenemedi ('{raw_lower[:60]}...'), varsayılan kullanılıyor: **{topic.upper()}**")
        else:
            st.info(f"Yönlendirilen Bilgi Alanı: **{topic.upper()}**")

        retriever = vector_stores[topic].as_retriever(search_kwargs={"k": 5})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        # 2. Adım: Cevap üretme
        result = qa_chain.invoke({"query": query})
        return result['result'], result['source_documents']

    except Exception as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            return "⚠️ **API Kotası Doldu:** Ücretsiz katman limitlerine ulaştınız. Lütfen 30-60 saniye bekleyip tekrar deneyin.", []
        return f"❌ Bir hata oluştu: {e}", []

# --- D. ARAYÜZ (UI) AYARLARI ---

LANG_OPTIONS = {
    'TR': {
        'title': "🤖 Çok Dilli Kodlama Asistanı",
        'intro': "Merhaba! Hangi teknik konuda yardımcı olabilirim?",
        'input_hint': "SQL JOIN hatası veya Python metodları hakkında sorun...",
        'expander': "📚 Kaynakları Gör",
        'spinner': "Cevap sentezleniyor..."
    },
    'EN': {
        'title': "🤖 Multilingual Coding Assistant",
        'intro': "Hello! How can I assist you with a technical problem?",
        'input_hint': "Ask about SQL JOIN errors or Python methods...",
        'expander': "📚 View Sources",
        'spinner': "Synthesizing answer..."
    }
}

if 'lang' not in st.session_state: st.session_state.lang = 'TR'

st.set_page_config(page_title="GenAI Coding Assistant", layout="wide")

if st.toggle("Türkçe / English", value=(st.session_state.lang == 'EN')):
    st.session_state.lang = 'EN'
else:
    st.session_state.lang = 'TR'

LANG = LANG_OPTIONS[st.session_state.lang]
st.title(LANG['title'])

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": LANG['intro']}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input(LANG['input_hint']):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(LANG['spinner']):
            response, sources = get_response(prompt)
            st.markdown(response)
            
            if sources:
                with st.expander(LANG['expander']):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Kaynak {i+1}:** {doc.metadata.get('source_dataset', 'Unknown')}")
                        st.code(doc.page_content[:300] + "...")
                        if i < len(sources)-1: st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
