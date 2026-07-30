import os
import json
import numpy as np
import dashscope
from dashscope import TextEmbedding
import chromadb

# 配置 API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 定义 embedding 函数
def get_embedding(text):
    resp = TextEmbedding.call(
        model='qwen3.7-text-embedding',
        input=text
    )
    return resp.output['embeddings'][0]['embedding']

# 余弦相似度
def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 连接 ChromaDB
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
chroma_path = os.path.join(data_dir, "chroma_db")
client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_collection(name="coffee_rag")

# 加载产品数据（用于精确匹配筛选）
json_path = os.path.join(data_dir, 'coffee_products.json')
with open(json_path, 'r', encoding='utf-8') as f:
    products = json.load(f)

# ============================================================
# 混合检索核心函数
# ============================================================

# 精确匹配关键词配置
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
        "match": None,  # 特殊处理：按价格排序
    },
}

# 冲泡关键词 -> 标准写法映射
BREWING_ALIAS = {
    "手冲": "手冲", "冷萃": "冷萃", "美式": "美式", "拿铁": "拿铁",
    "意式浓缩": "意式浓缩", "冰滴": "冰滴", "卡布奇诺": "卡布奇诺",
}

def find_exact_matches(query):
    """在查询中检测所有精确匹配关键词，返回匹配的过滤条件列表"""
    filters = []
    for category, config in PRECISE_FILTERS.items():
        for kw in config["keywords"]:
            if kw in query:
                filters.append({"category": category, "keyword": kw, "config": config})
                break  # 每个类别只取第一个匹配
    return filters

def filter_products(products, filters):
    """根据过滤条件筛选产品"""
    candidates = list(products)
    for f in filters:
        category = f["category"]
        kw = f["keyword"]
        config = f["config"]

        if category == "price":
            # 价格类：按价格排序
            if kw in ["平价", "便宜"]:
                candidates = sorted(candidates, key=lambda p: p["price"])
            elif kw == "贵":
                candidates = sorted(candidates, key=lambda p: p["price"], reverse=True)
            # "性价比" 不改变排序，靠向量语义
            continue

        # 其他类别：精确过滤
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

def hybrid_search(query, n_results=3):
    """混合检索：精确匹配 + 向量相似度排序"""
    query_vec = get_embedding(query)

    # 1. 检测精确匹配关键词
    filters = find_exact_matches(query)

    # 2. 过滤候选产品
    candidates = filter_products(products, filters)

    # 3. 如果没有候选，退化为纯向量检索
    if not candidates:
        print("⚠️ 精确匹配无结果，使用纯向量检索")
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=n_results
        )
        output = []
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            similarity = 1 - results['distances'][0][i]
            output.append((meta, similarity, doc))
        return output

    # 4. 对候选产品用向量相似度排序
    scored = []
    for p in candidates:
        doc = p.get("description", "")
        doc_vec = get_embedding(doc)
        sim = cosine_similarity(query_vec, doc_vec)
        scored.append((p, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n_results]

    output = []
    for p, sim in top:
        output.append((p, sim, p.get("description", "")))
    return output


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    test_queries = [
        "冷萃",
        "低酸度 手冲",
        "深度烘焙 拿铁",
        "高酸度 冰滴",
        "性价比高的美式",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 查询: \"{query}\"")
        print(f"{'='*60}")
        results = hybrid_search(query, n_results=3)
        for i, (meta, sim, doc) in enumerate(results):
            print(f"  {i+1}. {meta['name']} | 酸度: {meta.get('acidity_level','?')} | 价格: ¥{meta['price']} | 相似度: {sim:.4f}")
            print(f"     适合: {meta.get('suitable_for', [])}")
        print()
