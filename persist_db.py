import os
import sys
from dotenv import load_dotenv
import textwrap
from IPython.display import display
from IPython.display import Markdown

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# 사용자 정의 임베딩 클래스 정의
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def __call__(self, input):
        # 입력은 텍스트 리스트여야 합니다. 이를 임베딩합니다.
        return super().__call__(input)


# 사용자 정의 임베딩 함수 사용
embeddings = CustomOpenAIEmbeddings(api_key=api_key)

# Chroma DB 설정
vector_db_path = "/home/user/khtml-ai-llm/vectordb"

pdf_directory = "/home/user/khtml-ai-llm-tt/data/academic resources"

def load_pdf_files(pdf_directory):
    documents = []
    print("Start to load PDFs.")
    pdf_files = glob(os.path.join(pdf_directory, '*.pdf'))
    print(f"Found {len(pdf_files)} PDF files.")
    for pdf_file in pdf_files:
        print(f"Loading PDF file: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        print(f"Loaded {len(pdf_documents)} documents from {pdf_file}")
        documents.extend(pdf_documents)
    print(f"Loaded {len(documents)} PDF documents.")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunk_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def save_db(chunks, embeddings, vector_db_path):
    if not chunks:
        print("No chunks to save to VectorDB.")
        return None
    print("Save VectorDB")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=vector_db_path)
    vectordb.add_documents(documents=chunks)
    vectordb.persist()
    return vectordb.as_retriever()

# PDF 파일 로드
documents = load_pdf_files(pdf_directory)

# 문서 분할
chunks = split_documents(documents)

# 데이터베이스에 저장
retriever = save_db(chunks, embeddings, vector_db_path)

print("모든 PDF 파일이 벡터 데이터베이스에 저장되었습니다.")