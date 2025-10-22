# GenAI_Coding_Assistant_RAG
# Çok Dilli Kodlama ve Hata Ayıklama Asistanı Chatbotu
# English follows Turkish

## 1. Projenin Amacı

Bu chatbotun temel amacı, yazılım mühendisliği öğrencilerine ve geliştiricilere **Python, SQL ve Java** gibi dillerde **kodlama, syntax ve hata giderme** konularında anlık, güvenilir ve adımlı rehberlik sunmaktır. Proje, çalışan verilere göre ölçeklendirilmiş, iki katmanlı bir RAG mimarisiyle, bilmediği konuları reddedebilen bir mentor rolünü üstlenir.

### Elde Edilen Sonuçlar
* **Çok Dilli Kod Üretimi:** Python, SQL, Java dillerinde çalışan kod örnekleri sunabilmektedir.
* **Dil Bağımsızlığı ve Çıktı (Polylingual Support):** Arayüze eklenen dil değiştirme butonu sayesinde, **Türkçe ve İngilizce arayüz** desteği sunulmaktadır. Ayrıca, chatbot sorgu diline göre farklı dillerde de cevap verebilmektedir. **Japonca ve İspanyolca gibi farklı dillerde bile işlevsel kod üretebildiği test edilmiştir.**
* **Akıllı Yönlendirme (Routing):** Sorguları, Gemini tarafından sınıflandırarak doğru bilgi tabanına (`kodlama_rehberi` veya `hata_ayiklama_asistanı`) yönlendirir.
* **Hata Ayıklama Desteği:** Gerçek yazılım sorunları içeren veri setlerinden beslenerek yaygın hataların nedenlerini ve çözüm yollarını açıklar.

## 2. Veri Seti Hakkında Bilgi

Proje, iki ana bilgi tabanını beslemek için **Hugging Face** platformundaki lisanslı ve **çalıştığı doğrulanmış** veri setlerinden programatik olarak veri çekmektedir.

**Metodoloji ve Kaynaklar:**
* Veri setleri, LangChain'in araçları kullanılarak programatik olarak çekilir ve lokal ChromaDB'de indekslenir.

| Bilgi Tabanı | Kaynaklar | Sütun Adı |
| :--- | :--- | :--- |
| **Kodlama Rehberi** | `TokenBender/code_instructions_122k_alpaca_style`, `b-mc2/sql-create-context`, `red1xe/code_instructions` | `instruction` |
| **Hata Ayıklama Asistanı** | `devangb4/scikit-learn-issues` | `text` |

## 3. Kullanılan Yöntemler ve Teknolojiler

| Kategori | Teknoloji | Amaç |
| :--- | :--- | :--- |
| **Mimari** | Çok Katmanlı RAG (Routing tabanlı) | Kullanıcı niyetine göre doğru bilgi tabanına yönlendirme. |
| **LLM (Generation)** | Gemini API (gemini-2.5-flash) | Yüksek hızlı ve bağlamsal zengin cevap üretimi. |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | **Yerel model kullanımı** (API Embedding limitlerini aşmak için). |
| **RAG Framework** | LangChain & ChromaDB | Veri yönetimi, parçalama (chunking) ve vektör tabanlı depolama. |
| **Web Arayüzü** | Streamlit | Hızlı prototipleme ve interaktif sohbet arayüzü. |

## 4. Kodun Çalışma Kılavuzu

Bu adımlar, projenin tüm bağımlılıklarını kurmak ve indekslemeyi başlatmak için gereklidir.

### 4.1. Ortamın Hazırlanması

| İşlem | Linux / macOS (Bash) | Windows (PowerShell / CMD) |
| :--- | :--- | :--- |
| **1. Sanal Ortam Oluşturma** | `python3 -m venv venv` | `python -m venv venv` |
| **2. Sanal Ortamı Aktifleştirme** | `source venv/bin/activate` | `.\venv\Scripts\activate` |
| **3. Bağımlılıkları Kurma** | `pip install -r requirements.txt` | `pip install -r requirements.txt` |

### 4.2. Güvenlik ve Veritabanı Oluşturma

1.  **API Anahtarı:** Projenin ana dizininde **`.env`** dosyası oluşturulmalı ve `GEMINI_API_KEY` veya `GOOGLE_API_KEY` buraya eklenmelidir.
2.  **İndeksleme (Veritabanı Oluşturma):** Sanal ortam aktifken, Hugging Face'den verileri çeker ve yerel ChromaDB indekslerini oluşturur:
    ```bash
    python3 rag_pipeline.py
    ```
    *(Windows'ta `python` komutunu kullanın: `python rag_pipeline.py`)*

### 4.3. Web Arayüzünün Çalıştırılması

İndeksler oluştuktan sonra, arayüz başlatılır:

```bash
streamlit run app.py
```
Tarayıcınızda otomatik olarak açılacaktır.

Ayrıca bu link üzerinden web uygulamasına doğrudan erişilebilir: https://genaicodingassistantrag-26mwl2wh4fhtb2qfdue4nf.streamlit.app/ 


# GenAI_Coding_Assistant_RAG
# Multilingual Coding and Debugging Assistant Chatbot

## 1. Project Objective and Outcomes

The core objective of this chatbot is to provide prompt, reliable, and step-by-step guidance to software engineering students and developers on **coding, syntax, and troubleshooting** across languages such as **Python, SQL, and Java**. The project employs a two-layered RAG architecture, scaled with verified, functional data, acting as a mentor that can intelligently reject questions outside its scope.

### Key Verified Outcomes
* **Multilingual Code Generation:** It can generate working code examples in Python, SQL, and Java based on commands given in Turkish, English, and other languages.
* **Polylingual Support (Language Independence and Output):** The interface offers **Turkish and English support** via a language toggle button. Furthermore, the chatbot can generate responses in the language of the query, and has been successfully tested to produce functional code even in languages like **Japanese and Spanish**.
* **Intelligent Routing:** Classifies queries using a Gemini-powered system, directing the request to the correct knowledge base (`kodlama_rehberi` coding guide or `hata_ayiklama_asistanı` debugging assistant).
* **Debugging Assistance:** Provides solutions for common errors like `KeyError` or `TypeError` by utilizing datasets containing real-world software issues (GitHub Issues).

## 2. Data Set Methodology and Sources

The project feeds its two main knowledge bases by programmatically pulling data from **verified and licensed** Hugging Face datasets.

**Methodology and Sources:**
* Data sets are pulled using LangChain tools and indexed in a local ChromaDB instance.

| Knowledge Base | Sources | Text Column |
| :--- | :--- | :--- |
| **Coding Guide** | `TokenBender/code_instructions_122k_alpaca_style`, `b-mc2/sql-create-context`, `red1xe/code_instructions` | `instruction` |
| **Debugging Assistant** | `devangb4/scikit-learn-issues` | `text` |

## 3. Used Methods and Technologies

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Architecture** | Multi-Layered RAG (Router-based) | Directs user intent to the optimal knowledge source. |
| **LLM (Generation)** | Gemini API (gemini-2.5-flash) | High-speed and contextually rich response generation. |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | **Local model usage** (to bypass API Embedding limits). |
| **RAG Framework** | LangChain & ChromaDB | Data management, chunking, and vector-based storage. |
| **Web Interface** | Streamlit | Rapid prototyping and interactive chat UI. |

## 4. Code Execution Guide

### 4.1. Environment Setup (Virtual Environment)

| Step | Linux / macOS (Bash) | Windows (PowerShell / CMD) |
| :--- | :--- | :--- |
| **1. Create Environment** | `python3 -m venv venv` | `python -m venv venv` |
| **2. Activate Environment** | `source venv/bin/activate` | `.\venv\Scripts\activate` |
| **3. Install Dependencies** | `pip install -r requirements.txt` | `pip install -r requirements.txt` |

### 4.2. Security and Database Creation

1.  **API Key:** An **`.env`** file must be created in the main directory containing `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
2.  **Indexing (Database Creation):** With the virtual environment active, run the indexing process:
    ```bash
    python3 rag_pipeline.py
    ```
    *(For Windows, use: `python rag_pipeline.py`)*

### 4.3. Starting the Web Interface

Once indexing is complete, start the interface:
```bash
streamlit run app.py
```
It will automatically start on your browser

Also the live web application can be accessed directly via this link: https://genaicodingassistantrag-26mwl2wh4fhtb2qfdue4nf.streamlit.app/
