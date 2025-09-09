# conda activate /secure/shared_data/rag_embedding_model/nvembed
# Standard library
import json
import logging
import os
import re
from typing import Any, Dict, List

# Third-party
import chromadb
import matplotlib.pyplot as plt
import pandas as pd
import pymupdf
import torch
import torch.nn.functional as F
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
from torch.nn import DataParallel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ------------------
# Logging Setup
# ------------------
os.makedirs("log", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # logging.FileHandler('log/0410_MA_3_probs_parallel_static.log', mode='w'), 
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger(__name__)
# ------------------
# Environment Setup
# ------------------
def setup_environment() -> None:
    """
    Loads environment variables and configures CUDA usage.
    """
    env_path = find_dotenv()
    if not env_path:
        env_path = "/home/yl3427/.env"  # fallback path
    if not load_dotenv(env_path):
        raise Exception("Failed to load .env file")

    # Adjust GPU environment as needed
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    logger.info("Environment setup complete.")

# --------------------
# Text Loading and Processing, Creating Documents
# --------------------
def clean_extra_whitespace_within_paragraphs(text):
    return re.sub(r'[ \t]+', ' ', text)

def group_broken_paragraphs(text):
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # text = re.sub(r"\n{2,}", "\n", text)
    return text

def create_documents(
    files,
    tokenizer,
    max_length: int = 512
) -> List[Document]:

    if not isinstance(files, list):
        files = [files]  

    documents = []
    for file_path in files:
        doc = pymupdf.open(file_path)
        text = ""
        
        print(f"{len(doc)} pages found in {file_path}")
        for page in doc:
            text += page.get_text()

        text = group_broken_paragraphs(text)
        text = clean_extra_whitespace_within_paragraphs(text)

        document = Document(
            page_content=text,
            metadata={"source": file_path}
        )
        documents.append(document)

    """
    Splits clinical text into smaller chunks using a tokenizer-based splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        separators=["\n\n", "\n", '(?<=[.?"\s])\s+'],
        tokenizer=tokenizer,
        chunk_size=max_length,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        is_separator_regex=True
    )
    docs_processed = (text_splitter.split_documents([doc]) for doc in documents)

    unique_texts = set()
    docs_processed_unique = []
    for doc_chunk in docs_processed:
        for doc in doc_chunk:
            if doc.page_content not in unique_texts:
                unique_texts.add(doc.page_content)
                docs_processed_unique.append(doc)

    return docs_processed_unique
# --------------------
# Embedding in Chroma
# --------------------
def embed_docs_in_chroma(
    docs: List[Document],
    embedding_model,
    collection,
    max_length: int = 1024
) -> None:
    """
    Embeds documents into the Chroma collection.
    """
    pbar = tqdm(total=len(docs), desc="Embedding Documents")
    for doc in docs:
        doc_text = doc.page_content
        doc_meta = doc.metadata
        doc_id = str(doc.metadata["start_index"])
        
        # Log each doc ID as we process it
        logger.info(f"Embedding doc_id={doc_id}...")

        with torch.no_grad():
            embeddings = embedding_model.encode(
                [doc_text],
                instruction="",
                max_length=max_length
            )
            embeddings = embeddings.cpu().numpy().tolist()

        collection.add(
            embeddings=embeddings,
            documents=[doc_text],
            metadatas=[doc_meta],
            ids=[doc_id],
        )
        pbar.update(1)
        torch.cuda.empty_cache()
    pbar.close()

    logger.info("All documents embedded and added to Chroma.")


# --------------------
# Main Embedding Flow
# --------------------
def main():
    """
    1) Setup environment
    2) Create documents from PDF
    3) Connect to Chroma
    4) Embed docs into Chroma
    """
    setup_environment()

    # 1) Paths
    chroma_db_path = "/secure/shared_data/rag_embedding_model/chroma_db"
    model_cache_dir = "/secure/shared_data/rag_embedding_model"
    model_name = "nvidia/NV-Embed-v2"

    # 2) Create Documents
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_cache_dir
    )
    docs_processed = create_documents("ajcc_7thed_cancer_staging_manual.pdf", tokenizer)
  

    # 3) Connect to Chroma
    client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(allow_reset=True)
    )
    collection = client.get_or_create_collection(
        name="ajcc",
        metadata={"hnsw:space": "cosine"}
    )

    # 4) Load Embedding Model & Embed Docs
    embedding_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_cache_dir,
        device_map="auto"
    )
    embed_docs_in_chroma(docs_processed, embedding_model, collection)

    logger.info("Embedding script completed successfully.")


    # 5) Querying
    cancer_type_map = {'BLCA': 'Bladder Urothelial Carcinoma',
        'HNSC': 'Head and Neck Squamous Cell Carcinoma',
        'STAD': 'Stomach Adenocarcinoma',
        'CESC': 'Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma',
        'KIRC': 'Kidney Renal Clear Cell Carcinoma',
        'PRAD': 'Prostate Adenocarcinoma',
        'KIRP': 'Kidney Renal Papillary Cell Carcinoma',
        'KICH': 'Kidney Chromophobe',
        'LIHC': 'Liver Hepatocellular Carcinoma',
        'BRCA': 'Breast Invasive Carcinoma',
        'LUAD': 'Lung Adenocarcinoma',
        'PAAD': 'Pancreatic Adenocarcinoma',
        'THCA': 'Thyroid Carcinoma',
        'MESO': 'Mesothelioma',
        'ACC': 'Adrenocortical Carcinoma',
        'CHOL': 'Cholangiocarcinoma',
        'TGCT': 'Testicular Germ Cell Tumors',
        'LUSC': 'Lung Squamous Cell Carcinoma',
        'READ': 'Rectum Adenocarcinoma',
        'SKCM': 'Skin Cutaneous Melanoma',
        'COAD': 'Colon Adenocarcinoma',
        'UVM': 'Uveal Melanoma',
        'ESCA': 'Esophageal Carcinoma'}

    for key, value in cancer_type_map.items():
        print(f"{key}: {value}")

        query_t14 = f"A list of rules as knowledge that help predict the T stage for {value}"
        query_n03 = f"A list of rules as knowledge that help predict the N stage for {value}"
        queries = [query_t14, query_n03]
        query_prefix = "Instruct: Retrieve passages that define the following cancer staging rules:\n"
    
        query_embeddings = embedding_model.encode(queries, instruction=query_prefix, max_length=1024).detach().cpu().numpy() # .detach().cpu().numpy().tolist()

        results = collection.query(query_embeddings=query_embeddings,
                                include=["documents", "distances"],
                                n_results=5)

        rag_raw_t14 = '\n'.join(results['documents'][0])
        rag_raw_n03 = '\n'.join(results['documents'][1])

        variables = {
            "rag_raw_t14": rag_raw_t14,
            "rag_raw_n03": rag_raw_n03,
        }

        with open(f'context/context_{key}.json', 'w') as json_file:
            json.dump(variables, json_file, indent=4)

    logger.info("Context files created successfully.")


if __name__ == "__main__":
    main()
