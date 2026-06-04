"""FastAPI推理接口"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="MediGuard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from agent.graph import build_graph
        _graph = build_graph()
    return _graph


class QueryRequest(BaseModel):
    user_input: str
    chat_history: Optional[List[str]] = []


class QueryResponse(BaseModel):
    department: str
    urgency: str
    should_see_doctor: bool
    is_safe: bool
    safety_reason: str
    final_response: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="请输入症状描述")

    graph = get_graph()
    result = graph.invoke({
        "user_input": req.user_input,
        "retry_count": 0,
        "chat_history": req.chat_history,
    })

    return QueryResponse(
        department=result.get("department", "全科"),
        urgency=result.get("urgency", "medium"),
        should_see_doctor=result.get("should_see_doctor", False),
        is_safe=result.get("is_safe", True),
        safety_reason=result.get("safety_reason", ""),
        final_response=result.get("final_response", ""),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)