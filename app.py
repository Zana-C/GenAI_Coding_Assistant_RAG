# app.py - ÇOK DİLLİ KODLAMA ASİSTANI

import streamlit as st
from rag_pipeline import initialize_rag_system
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA

# --- A. BAŞLANGIÇ AYARLARI VE CACHING ---
# RAG sistemini sadece bir kere yüklemek için st.cache_resource kullanılır
@st.cache_resource
def setup_system():
    # rag_pipeline'dan vektör depolarını ve LLM'i yükler
    try:
        # initialize_rag_system'dan sadece stores ve llm alınıyor
        _, vector_stores, llm = initialize_rag_system()
        return vector_stores, llm
    except Exception as e:
        st.error(f"RAG Sistemi Başlatılamadı! Lütfen rag_pipeline.py dosyasını kontrol edin. Detay: {e}")
        st.stop()

# Vektör depolarını ve LLM modelini yükle
vector_stores, llm = setup_system()


# --- B. YÖNLENDİRME (ROUTING) MANTIĞI VE RAG ZİNCİRİ ---

# Kullanıcının sorgusunun hangi bilgi tabanına gideceğini belirleyen prompt.
CLASSIFIER_PROMPT = """
Sen, bir sorguyu iki kategoriden birine sınıflandıran bir yapay zeka modelisin.
Görevin, sadece ve sadece uygun kategori adını (kodlama_rehberi, hata_ayiklama_asistani) döndürmektir. Başka hiçbir metin veya açıklama ekleme.

Kategoriler:
1. kodlama_rehberi: Python, SQL veya JavaScript syntax'ı, temel fonksiyonları, kod parçacığı üretimi ve kavramsal kodlama bilgisi hakkındaki sorular.
2. hata_ayiklama_asistani: Kod hataları (error), yazılım sorunları (bug), hata mesajlarının çözümü ve genel sorun giderme (troubleshooting) hakkında sorular.

Sorgu: "{query}"
Cevap (Sadece kategori adı):
"""

# Gemini'yi kullanarak sorguyu sınıflandırma zincirini kurar
classifier_chain = (
    PromptTemplate.from_template(CLASSIFIER_PROMPT)
    | llm
    | StrOutputParser()
)

# get_response fonksiyonu, Streamlit arayüzünde çağrılmadan önce tanımlanmalıdır
def get_response(query: str):
    # 1. Sorguyu Sınıflandırma
    topic = classifier_chain.invoke({"query": query}).strip().lower()

    if topic in vector_stores:
        # Sınıflandırma bilgisini Streamlit mesajında gösteriyoruz
        st.info(f"Yönlendirilen Bilgi Alanı: **{topic.upper()}**")
        retriever = vector_stores[topic].as_retriever(search_kwargs={"k": 5})

        # 2. Seçilen alana özel RAG zincirini kurma
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # 3. Cevabı Alma
        result = qa_chain({"query": query})

        # 4. Yanıtı ve Kaynakları Temizleme
        return result['result'], result['source_documents']
    else:
        # Eğer sınıflandırma başarısız olursa
        return f"Üzgünüm, sorgunuzu sınıflandıramadım. Konunuzun **{list(vector_stores.keys())}** alanlarından birine ait olduğundan emin olun. Sınıflandırma sonucu: {topic}", []


# --- C. YARDIMCI DİL VE ARAYÜZ AYARLARI ---

LANG_OPTIONS = {
    'TR': {'page_title': "Çok Dilli Kod Asistanı", 'title': "🤖 Çok Dilli Kodlama ve Hata Ayıklama Asistanı",
           'markdown': "Bir Yazılım Mühendisi mentorunuz olarak, **Python, Java, SQL** gibi dillerde **kodlama, syntax ve hata giderme** konularındaki sorularınıza rehberlik edeceğim. **Lütfen konuyu belirterek soru sorun.**",
           'input_hint': "Python liste metotları, SQL JOIN hatası veya Java kodu hakkında bir soru sorun...",
           'intro': "Merhaba! Nasıl bir teknik konuda yardımcı olabilirim?",
           'expander': "📚 Kaynakları Gör (RAG Doğrulama)",
           'spinner': "Cevap aranıyor ve sentezleniyor..."},

    'EN': {'page_title': "Multilingual Coding Assistant", 'title': "🤖 Multilingual Coding & Debugging Assistant",
           'markdown': "As a Software Engineer mentor, I will guide you on **coding, syntax, and debugging** issues in languages like **Python, Java, and SQL**. **Please specify the topic in your query.**",
           'input_hint': "Ask a question about Python list methods, an SQL JOIN error, or a Java program...",
           'intro': "Hello! How can I assist you with a technical problem?",
           'expander': "📚 View Sources (RAG Verification)",
           'spinner': "Searching and synthesizing the answer..."}
}

# Dil seçimi durumu (state)
if 'lang' not in st.session_state:
    st.session_state.lang = 'TR'

# --- D. STREAMLIT ANA ARAYÜZÜ ---

# Dil değiştirme butonu
st.set_page_config(page_title=LANG_OPTIONS[st.session_state.lang]['page_title'], layout="wide")
col1, col2 = st.columns([0.8, 0.2])
with col2:
    # st.toggle, dil durumunu st.session_state'te tutar
    if st.toggle("Türkçe / English", value=(st.session_state.lang == 'EN')):
        st.session_state.lang = 'EN'
    else:
        st.session_state.lang = 'TR'

# Güncel dil ayarını al
LANG = LANG_OPTIONS[st.session_state.lang]

# Başlık ve Açıklamalar
st.title(LANG['title'])
st.markdown(LANG['markdown'])

# Chat geçmişini başlatma
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": LANG['intro']}]

# Geçmiş mesajları görüntüle
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcı girişi
if prompt := st.chat_input(LANG['input_hint']):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Cevap üretimi
    with st.chat_message("assistant"):
        with st.spinner(LANG['spinner']):
            response, sources = get_response(prompt)

            # Cevabı görüntüle
            st.markdown(response)

            # Kaynak Gösterimi AÇILIR PENCERE (Expander)
            if sources:
                with st.expander(LANG['expander'], expanded=False):
                    for i, doc in enumerate(sources):
                        # Kaynak gösterimi dili, arayüz diline göre dinamik olarak değişir
                        if st.session_state.lang == 'EN':
                            source_info = f"**Source {i+1}** - Dataset: `{doc.metadata.get('source_dataset', 'Unknown')}`"
                        else:
                            source_info = f"**Kaynak {i+1}** - Veri Seti: `{doc.metadata.get('source_dataset', 'Bilinmiyor')}`"

                        st.markdown(source_info)
                        st.code(doc.page_content[:400] + "...")
                        if i < len(sources) - 1:
                            st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
