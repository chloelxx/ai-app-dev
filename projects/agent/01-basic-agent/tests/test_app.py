from fastapi.testclient import TestClient

from src.main import app
from src.api import routes


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_calc_tool(monkeypatch):
    # 避免真实调用大模型，这里只测试 calc: 工具分支

    # 确保使用的是原始路由中的 agent_service，但我们不需要改它，
    # 因为 calc: 分支不依赖 LLM。
    resp = client.post(
        "/agent/chat",
        json={"message": "calc: 1+2*3"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "1+2*3" in data["reply"]
    assert "7" in data["reply"]


