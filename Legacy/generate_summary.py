# generate_summary.py

from rag_chain import initialize_rag_chain

# 요약 생성 함수
def generate_summary_gpt4o(transcription_text: str, api_key: str) -> str:
    # RAG Chain 초기화
    rag_chain = initialize_rag_chain(
        api_key=api_key,
        persist_directory="/home/user/khtml-ai-llm-tt/vectordb",
        model_name="gpt-4o-mini"
    )

    # RAG Chain을 사용해 답변을 생성
    response = rag_chain.invoke({
        'question': f"The audio transcription is: {transcription_text}",
        'context': "Transcript summary task"
    })
    return response.strip()
