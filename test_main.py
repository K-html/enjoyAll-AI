from fastapi.testclient import TestClient
from main import app  # main.py 파일에서 FastAPI 애플리케이션 가져오기

client = TestClient(app)

def test_handle_text():
    # 테스트할 JSON 데이터
    test_data = {"text": "This is a test text"}
    
    # POST 요청을 보내고 응답을 받습니다
    response = client.post("/text", json=test_data)
    
    # 응답 상태 코드와 데이터를 확인합니다
    assert response.status_code == 200
    assert "summary" in response.json()
    assert response.json()["summary"] == "Summarized text"  # 실제 요약 로직에 맞게 수정 필요
