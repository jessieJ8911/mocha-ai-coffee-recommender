from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from functools import lru_cache
import sys
sys.path.append('..')  # 方便引用上层模块

# 假设你已经把检索逻辑封装成了函数
from core.retrieval import hybrid_search

app = FastAPI(
    title="Mocha AI 智能咖啡推荐系统",
    description="基于RAG的语义检索+混合排序推荐引擎",
    version="1.0.0"
)

# 定义请求体结构
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

# 定义响应体结构
class SearchResult(BaseModel):
    name: str
    price: float
    acidity: str
    suitable_for: List[str]
    similarity: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.get("/")
def read_root():
    return {"message": "Mocha AI 咖啡推荐系统已启动 🚀", "docs": "/docs"}

@app.post("/search", response_model=SearchResponse)
def search_coffee(request: SearchRequest):
    """
    智能咖啡检索接口
    - 支持语义搜索（如"酸度低"、"巧克力味"）
    - 支持精确过滤（如"冷萃"、"手冲"）
    - 支持组合条件（如"高酸度 冰滴"）
    """
    try:
        results = hybrid_search(request.query, n_results=request.top_k)
        return SearchResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 额外加一个健康检查接口
@app.get("/health")
def health_check():
    return {"status": "healthy", "total_products": 30}

# 智能推荐语接口
import os
import json
from dashscope import Generation

# 加载完整产品数据（用于获取 flavor_notes、roast_level 等补充字段）
@lru_cache()
def _load_raw_products():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    json_path = os.path.join(data_dir, "coffee_products.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)}

@app.post("/recommend_with_reason")
def recommend_with_reason(request: SearchRequest):
    """
    智能推荐 + 大模型生成推荐语
    返回 top_k 条结果，仅第 1 名附带 LLM 生成的推荐理由
    """
    # 1. 先调用混合检索，取 top_k 条
    results = hybrid_search(request.query, n_results=request.top_k)
    
    if not results:
        return {
            "query": request.query,
            "results": [],
            "reason": "没有找到合适的咖啡，试试其他关键词吧"
        }
    
    # 2. 仅为第 1 名生成推荐语
    top1 = results[0]
    all_products = _load_raw_products()
    detail = all_products.get(top1.get("id", ""), {})
    
    prompt = f"""你是一位专业的咖啡推荐师，语气亲切自然，像在和熟客聊天。

用户说：「{request.query}」

你为他推荐了这款咖啡：
- 名称：{top1['name']}
- 产地：{detail.get('origin', '未知')}
- 烘焙度：{detail.get('roast_level', '未知')}
- 酸度：{top1['acidity']}
- 风味：{detail.get('flavor_notes', '风味独特')}
- 描述：{detail.get('description', '一款不错的咖啡')}

请用一段温暖、口语化的文字（80-120字），向用户解释为什么这款咖啡适合他/她。
要求：
- 不要复述参数，要说人话
- 要结合用户的需求来推荐
- 语气像咖啡店老板在跟熟客聊天
- 结尾给一个简单的冲煮建议（比如水温、适合什么器具）
"""

    try:
        response = Generation.call(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=250
        )
        reason = response.output.choices[0].message.content
    except Exception:
        flavor = detail.get("flavor_notes", "风味独特")
        desc = detail.get("description", "试试看哦～")
        reason = f"这款{top1['name']}{flavor}，{desc[:50]}... 建议试试看～"
    
    # 3. 给第 1 名注入推荐语
    results[0]["reason"] = reason
    
    return {
        "query": request.query,
        "results": results
    }