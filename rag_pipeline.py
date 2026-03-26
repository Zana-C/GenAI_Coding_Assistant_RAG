# rag_pipeline.py

import os
from dotenv import load_dotenv
from typing import List, Dict

# GEREKLİ KÜTÜPHANELER
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Ortam değişkenlerini (.env) yükle
load_dotenv()

# API anahtarını kontrol et ve gerekirse düzenle
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("HATA: GOOGLE_API_KEY veya GEMINI_API_KEY ortam değişkeni bulunamadı. Lütfen .env dosyanızı oluşturun.")

# --- A. SABİTLER VE KONFİGÜRASYON ---

CHROMA_DB_PATH = "chroma_db"

# Embedding Model Seçimi (Yerel model ile API limiti aşılır)
USE_LOCAL_EMBEDDINGS = True

if USE_LOCAL_EMBEDDINGS:

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Yerel (HuggingFace) model
else:

    EMBEDDING_MODEL = "models/embedding-001" # Google Gemini API modeli

LLM_MODEL = "gemini-2.0-flash" # Cevap üretimi için hızlı Gemini modeli



# RAG'ın bilgi çekeceği iki ana kategori (Çok Katmanlı Yapı)

DATASET_CONFIG = {
    "kodlama_rehberi": {
        "datasets": [
            {"name": "TokenBender/code_instructions_122k_alpaca_style", "column": "instruction"},
            {"name": "b-mc2/sql-create-context", "column": "question"},
            {"name": "red1xe/code_instructions", "column": "instruction"}
        ],
        "split_percentage": "3%"
    },
    "hata_ayiklama_asistani": {
        "datasets": [
            {"name": "devangb4/scikit-learn-issues", "column": "text"}
        ],
        "split_percentage": "15%"
    }
}

# --- B. VERİ İŞLEME VE İNDEKSLEME FONKSİYONLARI ---

def load_and_split_data(datasets_info: List[Dict], split_percentage: str) -> List[Document]:
    """
    TR: Hugging Face veri setlerini yükler, yapılandırılmış sütunlara göre dökümana çevirir ve parçalar.
    EN: Loads Hugging Face datasets, converts them into documents based on structured columns, and chunks them.
    """
    all_chunks = []

    # TR: Liste içindeki her bir veri seti konfigürasyonu için döngü başlat
    # EN: Start a loop for each dataset configuration in the list
    for ds_info in datasets_info:
        # TR: Sözlükten veri seti adını ve ilgili metin sütununu al
        # EN: Retrieve the dataset name and the relevant text column from the dictionary
        ds_name = ds_info["name"]
        text_column = ds_info["column"] 
        
        print(f"-> Veri Seti Yükleniyor / Loading Dataset: {ds_name} ({split_percentage})")

        # TR: Veri setini farklı split (ayrım) adlarına göre yüklemeyi dene
        # EN: Attempt to load the dataset using different split names
        try:
            # TR: Öncelikle 'train' (eğitim) setini belirlenen yüzde kadar yükle
            # EN: First, try loading the 'train' split with the specified percentage
            dataset = load_dataset(ds_name, split=f'train[:{split_percentage}]')
        except Exception:
            try:
                # TR: 'train' yoksa, tüm veriyi kapsayan 'all' split'ini dene
                # EN: If 'train' is missing, try the 'all' split
                dataset = load_dataset(ds_name, split=f'all[:{split_percentage}]')
            except Exception:
                # TR: Hiçbir split bulunamazsa hata mesajı ver ve bir sonrakine geç
                # EN: If no split is found, print error and continue to the next dataset
                print(f"  ❌ {ds_name} yüklenemedi / failed to load.")
                continue

        # TR: Ham veriyi LangChain 'Document' formatına dönüştür
        # EN: Transform raw data into LangChain 'Document' format
        documents = [
            # TR: Metni string'e çevir ve kaynağı metadata (üstveri) olarak ekle
            # EN: Convert text to string and add the source as metadata
            Document(page_content=str(row[text_column]), metadata={"source_dataset": ds_name})
            for row in dataset 
            # TR: Sütun kontrolü ve boş veri (None) kontrolü yaparak hataları önle
            # EN: Prevent errors by checking for column existence and None values
            if text_column in row and row[text_column] is not None
        ]

        print(f"  ✓ {len(documents)} doküman yüklendi / docs loaded (Sütun/Column: {text_column})")

        # TR: Büyük metinleri anlamlı parçalara ayırmak için splitter (ayırıcı) yapılandır
        # EN: Configure the splitter to break down large texts into meaningful chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,        # TR: Her parçanın karakter sınırı / EN: Character limit per chunk
            chunk_overlap=200,      # TR: Parçalar arası bağlamsal çakışma / EN: Contextual overlap between chunks
            separators=["\n\n", "\n", " ", ""] # TR: Ayırma öncelikleri / EN: Splitting priorities
        )
        
        # TR: Dokümanları parçala ve ana listeye ekle
        # EN: Split documents and extend the main list
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)

    # TR: Oluşturulan tüm parçaları (chunk) döndür
    # EN: Return all generated chunks
    return all_chunks



def setup_vector_stores() -> Dict[str, Chroma]:
    """
    Çok katmanlı RAG mimarisi için Vektör İndekslerini (ChromaDB) kurar veya yükler.

    """
    if USE_LOCAL_EMBEDDINGS:
        print("\n📥 Yerel embedding modeli kullanılıyor...")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Yerel embedding modeli hazır!\n")
        batch_size = 500
    else:
        print("\n🌐 Google Gemini embeddings kullanılıyor...")
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        print("✅ Gemini embeddings hazır!\n")
        batch_size = 100

    vector_stores = {}

    for topic, config in DATASET_CONFIG.items():
        db_path = os.path.join(CHROMA_DB_PATH, topic)

        # İndeks yoksa oluştur, varsa yükle
        if not os.path.exists(db_path) or not os.listdir(db_path):
            print(f"\n--- {topic.upper()} İndeksi YENİDEN Oluşturuluyor ---")

            # Veri yükleme ve parçalama
            chunks = load_and_split_data(
                config["datasets"],
                config["split_percentage"]
            )

            if not chunks:
                print(f"  ⚠ {topic} için hiç chunk oluşturulamadı, atlanıyor...")
                continue

            # ChromaDB'ye Batch Halinde İndeksleme
            BATCH_SIZE = batch_size
            print(f"  📦 {len(chunks)} chunk, {BATCH_SIZE}'lük batch'ler halinde işleniyor...")

            vectorstore = None
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

                print(f"  ⏳ Batch {batch_num}/{total_batches} işleniyor... ({len(batch)} chunk)")

                if vectorstore is None:
                    # İlk batch: Yeni vectorstore oluştur
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=db_path
                    )
                else:
                    # Sonraki batch'ler: Mevcut vectorstore'a ekle
                    vectorstore.add_documents(batch)


            print(f"✓ {topic.upper()} İndeksi Başarıyla Oluşturuldu ({db_path})")
        else:
            print(f"\n--- {topic.upper()} İndeksi Yükleniyor ---")
            # Önceden oluşturulmuş DB'yi yükle
            vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
            print(f"✓ {topic.upper()} İndeksi Yüklendi")

        vector_stores[topic] = vectorstore

    return vector_stores

# --- C. RAG SİSTEMİNİN BAŞLATILMASI ---

def initialize_rag_system():
    """
    RAG sistemini başlatır, Vektör Depolarını yükler ve LLM'i hazırlar.
    Bu, app.py tarafından çağrılır.
    """
    print("\n🚀 Çok Katmanlı RAG Sistemi Başlatılıyor...\n")

    # Adım 1: Vektör İndekslerini Kurma/Yükleme
    vector_stores = setup_vector_stores()

    if not vector_stores:
        raise Exception("Hiçbir vektör deposu oluşturulamadı!")

    # Adım 2: LLM'i Hazırlama (Gemini)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    # Varsayılan retriever (routing mantığı app.py'de yer alır)
    default_topic = list(vector_stores.keys())[0]
    default_retriever = vector_stores[default_topic].as_retriever(search_kwargs={"k": 4})

    default_rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=default_retriever,
        return_source_documents=True  # Kaynak gösterimi için
    )

    return default_rag_chain, vector_stores, llm


# --- D. UYGULAMA MANTIĞI ---

if __name__ == "__main__":
    try:
        # İndeksleme işlemini tetikler
        rag_chain, vector_stores, llm = initialize_rag_system()

        print("\n" + "="*60)
        print("✅ TÜM ChromaDB İNDEKSLERİ BAŞARIYLA HAZIRLANDI")
        print("="*60)
        print(f"\nHazır Vektör Depoları: {list(vector_stores.keys())}")
        print("\n💡 Artık Streamlit arayüzünü (app.py) çalıştırabilirsiniz!")

        # Test sorgusu (opsiyonel)
        print("\n--- Test Sorgusu ---")
        test_result = rag_chain.invoke({"query": "What is scikit-learn?"})
        print(f"Soru: What is scikit-learn?")
        print(f"Cevap: {test_result['result'][:200]}...")

    except Exception as e:
        print(f"\n❌ KRİTİK HATA: RAG Sistemi başlatılamadı.")
        print(f"Hata Detayı: {e}")
        print("\n🔍 Kontrol Listesi:")
        print("  1. GEMINI_API_KEY .env dosyasında tanımlı mı?")
        print("  2. Sanal ortam (venv) aktif mi?")
        print("  3. İnternet bağlantınız çalışıyor mu?")
        print("  4. requirements.txt dosyası kuruldu mu? (pip install -r requirements.txt)")
