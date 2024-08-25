
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import chromadb

# 사용자 정의 임베딩 클래스 정의
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def __call__(self, input):
        # 입력은 텍스트 리스트여야 합니다. 이를 임베딩합니다.
        return super().__call__(input)

# 사용자 정의 임베딩 함수 사용
embeddings = CustomOpenAIEmbeddings(api_key=api_key)

vectordb = Chroma(persist_directory=vector_db_path, 
                  embedding_function=embeddings)
retriever = vectordb.as_retriever()