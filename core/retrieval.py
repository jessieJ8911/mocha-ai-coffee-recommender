"""
混合检索模块
- 精确匹配：根据冲泡方式、酸度、烘焙程度、价格等关键词过滤
- 向量检索：用 ChromaDB 做语义相似度搜索
- 混合策略：先精确过滤，再向量排序；无匹配时回退纯向量检索
"""

from core.embedding import get_embedding, get_embeddings_batch, cosine_similarity
from core.database import get_collection, load_products

# ---- 精确匹配关键词配置 ----
PRECISE_FILTERS = {
    "brewing": {
        "keywords": ["手冲", "冷萃", "美式", "拿铁", "意式浓缩", "冰滴", "卡布奇诺"],
        "field": "suitable_for",
        "match": lambda val, kw: any(kw in item for item in val),
    },
    "acidity": {
        "keywords": ["高酸度", "中等酸度", "低酸度", "高酸", "低酸", "酸度高", "酸度低"],
        "field": "acidity_level",
        "match": lambda val, kw: kw in val,
    },
    "roast": {
        "keywords": ["浅度烘焙", "中度烘焙", "深度烘焙", "浅焙", "中焙", "深焙"],
        "field": "roast_level",
        "match": lambda val, kw: kw in val,
    },
    "price": {
        "keywords": ["平价", "便宜", "贵", "性价比"],
        "field": "price",
        "match": None,  # 特殊处理：按价格排序，不过滤
    },
}

# 冲泡关键词别名映射（便于扩展）
BREWING_ALIAS = {
    "手冲": "手冲", "冷萃": "冷萃", "美式": "美式", "拿铁": "拿铁",
    "意式浓缩": "意式浓缩", "冰滴": "冰滴", "卡布奇诺": "卡布奇诺",
}


def find_exact_matches(query: str) -> list[dict]:
    """在查询文本中检测精确匹配关键词

    Args:
        query: 用户查询字符串

    Returns:
        匹配到的过滤条件列表，每个条件包含 category/keyword/config
    """
    filters = []
    for category, config in PRECISE_FILTERS.items():
        for kw in config["keywords"]:
            if kw in query:
                filters.append({"category": category, "keyword": kw, "config": config})
                break  # 每个类别只取第一个匹配关键词
    return filters


def filter_products(products: list[dict], filters: list[dict]) -> list[dict]:
    """根据过滤条件筛选产品

    Args:
        products: 产品字典列表
        filters: 过滤条件列表

    Returns:
        筛选后的产品列表
    """
    candidates = list(products)

    for f in filters:
        category = f["category"]
        kw = f["keyword"]
        config = f["config"]

        if category == "price":
            # 价格类：只排序，不剔除任何产品
            if kw in ["平价", "便宜"]:
                candidates = sorted(candidates, key=lambda p: p["price"])
            elif kw == "贵":
                candidates = sorted(candidates, key=lambda p: p["price"], reverse=True)
            # "性价比" 不改变排序，靠向量语义
            continue

        # 其他类别：精确字段匹配过滤
        field = config["field"]
        match_fn = config["match"]
        new_candidates = []
        for p in candidates:
            val = p.get(field, "")
            if match_fn(val, kw):
                new_candidates.append(p)

        if new_candidates:
            candidates = new_candidates

    return candidates


def vector_search(collection, query_str: str, n_results: int = 3) -> list[tuple]:
    """纯向量检索（ChromaDB 语义搜索）

    Args:
        collection: ChromaDB 集合
        query_str: 查询文本
        n_results: 返回数量

    Returns:
        [(meta_dict, similarity, doc_text), ...]
    """
    query_vec = get_embedding(query_str)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
    )
    output = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        similarity = 1 - results["distances"][0][i]
        output.append((meta, similarity, doc))
    return output


def _build_result_dict(meta: dict, similarity: float) -> dict:
    """将检索结果转换为统一的字典格式，匹配 API 的 SearchResult 模型"""
    return {
        "id": meta.get("id", ""),
        "name": meta.get("name", ""),
        "price": meta.get("price", 0),
        "acidity": meta.get("acidity_level", ""),
        "suitable_for": meta.get("suitable_for", []),
        "similarity": round(similarity, 4),
    }


def hybrid_search(
    query: str,
    n_results: int = 3,
    products: list[dict] | None = None,
    collection=None,
) -> list[dict]:
    """混合检索：精确匹配过滤 + 向量相似度排序

    流程：
    1. 解析查询中的精确关键词（冲泡方式、酸度、烘焙、价格）
    2. 用关键词过滤候选产品
    3. 对候选产品用向量相似度排序
    4. 若无精确匹配，回退为纯向量检索

    Args:
        query: 用户查询字符串
        n_results: 返回结果数量
        products: 产品列表（可选，不传则自动加载）
        collection: ChromaDB 集合（可选，不传则自动获取）

    Returns:
        [{"name", "price", "acidity", "suitable_for", "similarity"}, ...]
    """
    # 自动获取依赖
    if products is None:
        products = load_products()
    if collection is None:
        collection = get_collection("coffee_rag")

    query_vec = get_embedding(query)

    # 1. 检测精确匹配关键词
    filters = find_exact_matches(query)

    # 2. 过滤候选产品
    candidates = filter_products(products, filters)

    # 3. 无候选 → 退化为纯向量检索
    if not candidates:
        raw = vector_search(collection, query, n_results)
        return [_build_result_dict(meta, sim) for meta, sim, _ in raw]

    # 4. 对候选产品用向量相似度排序（批量 embedding，一次 API 调用）
    descs = [p.get("description", "") for p in candidates]
    desc_vecs = get_embeddings_batch(descs)

    scored = []
    for i, p in enumerate(candidates):
        sim = cosine_similarity(query_vec, desc_vecs[i])
        scored.append((p, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n_results]

    return [_build_result_dict(p, sim) for p, sim in top]


if __name__ == "__main__":
    # 快速测试
    test_queries = [
        "冷萃",
        "低酸度 手冲",
        "深度烘焙 拿铁",
        "高酸度 冰滴",
        "性价比高的美式",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f'查询: "{query}"')
        print(f"{'='*60}")
        results = hybrid_search(query, n_results=3)
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['name']} | 酸度: {r['acidity']} | "
                  f"¥{r['price']} | 相似度: {r['similarity']}")
            print(f"     适合: {r['suitable_for']}")
        print()
