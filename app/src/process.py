import re
from typing import List, Dict
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
from langchain.docstore.document import Document
import PyPDF2
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class DocumentProcessing:
    def __init__(self, max_tokens: int = 300, overlap: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s:@\-\.]", "", text)
        return re.sub(r"\s+", " ", text).strip()  # normalize newlines

    def preprocess_text(self, text: str) -> List[str]:
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)  # tokenize into sentences

        cleaned = []
        for i in sentences:
            words = i.split()
            filtered = [
                word for word in words if word.lower() not in self.stop_words
            ]  # remove stop words
            sen = ' '.join(filtered)
            cleaned.append(sen)

        return cleaned

    def create_chunks(self, sentences: List[str]) -> List[str]:
        chunks = []
        curr_chunk = []
        curr_length = 0

        for sentence in sentences:
            sentence_tokens = sentence.split()
            sentence_length = len(sentence_tokens)

            if curr_length + sentence_length > self.max_tokens:
                chunks.append(curr_chunk)
                curr_chunk = curr_chunk[-self.overlap:]
                curr_length = len(' '.join(curr_chunk).split())

            curr_chunk.extend(sentence_tokens)
            curr_length += sentence_length

        if curr_chunk:
            chunks.append(curr_chunk)

        return [' '.join(chunk) for chunk in chunks]

    def process_document(self, content: str, filename: str) -> Dict[str, List[str]]:
        sentences = self.preprocess_text(content)
        chunks = self.create_chunks(sentences)
        return {filename: chunks}

def ingest_document(file_path: str, vectorstore=None):
    # Read the document
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    # Process the document
    doc_processor = DocumentProcessing()
    processed_doc = doc_processor.process_document(text, file_path)
    
    # Convert processed chunks into LangChain Documents
    documents = [Document(page_content=chunk) for chunk in processed_doc[file_path]]
    
    # Embed and index the documents
    embeddings = OpenAIEmbeddings()
    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents, embeddings)
    else:
        vectorstore.add_documents(documents)
    
    return vectorstore

def create_llama_index(directory_path: str):
    # Load documents using LlamaIndex's SimpleDirectoryReader
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    # Create an index using LlamaIndex
    index = VectorStoreIndex.from_documents(documents)
    
    return index
