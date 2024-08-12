# rag_chain.py
from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 벡터 DB 초기화 및 RAG Chain 구성 함수
def initialize_rag_chain(api_key: str, persist_directory: str, model_name: str):
    # 벡터 DB 초기화
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    # 프롬프트 템플릿 설정
    SYS_PROMPT = '''
    {context}에 맞게 답변해줘.
    JSON 형태로 답변이 나와야해.
    OUTPUT 형태는 :  을 참고해줘.
    '''
    template = SYS_PROMPT + '''
    사용자 입력 메세지에 잘 따라줘야해 : {question}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    # 모델 초기화
    model = ChatOpenAI(model='gpt-4o', temperature=0)
    # 문서 포맷팅 함수
    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    

    # RAG Chain 구성
    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain
