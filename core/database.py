"""
ChromaDB 连接与数据加载模块
- 初始化 ChromaDB 客户端（持久化存储）
- 读取产品数据和用户评价
- 拼接文本、生成向量、存入向量数据库
"""

import os
import json
import chromadb

from core.embedding import get_embedding

# ---- 路径工具 ----
def _get_data_dir() -> str:
    """获取 data/ 目录的绝对路径"""
    core_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(core_dir)
    return os.path.join(project_root, "data")


# ---- 数据库初始化 ----
def get_client():
    """获取 ChromaDB 持久化客户端"""
    data_dir = _get_data_dir()
    chroma_path = os.path.join(data_dir, "chroma_db")
    return chromadb.PersistentClient(path=chroma_path)


def get_collection(name: str = "coffee_rag"):
    """获取已有的集合（只读）"""
    client = get_client()
    return client.get_collection(name=name)


def reset_collection(name: str = "coffee_rag"):
    """删除旧集合并创建新集合（用于重新写入数据）"""
    client = get_client()
    try:
        client.delete_collection(name=name)
        print(f"旧集合 [{name}] 已删除，将重新创建")
    except Exception:
        pass
    return client.create_collection(name=name)


# ---- 数据加载 ----
def load_products():
    """加载咖啡产品数据"""
    data_dir = _get_data_dir()
    json_path = os.path.join(data_dir, "coffee_products.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reviews():
    """加载用户评价数据"""
    data_dir = _get_data_dir()
    reviews_path = os.path.join(data_dir, "reviews.json")
    with open(reviews_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_reviews_by_product(reviews: list[dict]) -> dict[str, list[dict]]:
    """将评价按 product_id 分组

    Args:
        reviews: 评价列表

    Returns:
        {product_id: [review, ...], ...}
    """
    grouped = {}
    for review in reviews:
        pid = review["product_id"]
        if pid not in grouped:
            grouped[pid] = []
        grouped[pid].append(review)
    return grouped


# ---- 构建嵌入文本 ----
def build_product_text(product: dict, reviews: list[dict] | None = None) -> str:
    """为单个产品拼接完整的嵌入文本（含用户评价）

    Args:
        product: 产品数据字典
        reviews: 该产品的评价列表（可选）

    Returns:
        拼接后的完整文本
    """
    text_parts = [
        f"产品名称：{product['name']}",
        f"产地：{product['origin']}",
        f"烘焙程度：{product['roast_level']}",
        f"酸度：{product['acidity_level']}",
        f"风味：{product['flavor_notes']}",
        f"适合冲泡方式：{'、'.join(product['suitable_for'])}",
        f"口感标签：{'、'.join(product['taste_tags'])}",
        f"适用场景：{product['user_scenario']}",
        f"价格：{product['price']}元/{product['unit']}",
        f"详细描述：{product['description']}",
    ]

    if reviews:
        text_parts.append("用户评价：")
        for r in reviews:
            text_parts.append(f"  - {r['user']}（{r['rating']}星）：{r['comment']}")

    return "。".join(text_parts)


def build_product_metadata(product: dict) -> dict:
    """构建产品元数据（存入 ChromaDB 的 metadatas）"""
    return {
        "id": product["id"],
        "name": product["name"],
        "roast_level": product["roast_level"],
        "acidity_level": product["acidity_level"],
        "price": product["price"],
        "category": product["category"],
    }


# ---- 向量化并存入数据库 ----
def load_and_embed_all():
    """主流程：读取数据 → 拼接文本 → 生成向量 → 存入 ChromaDB"""
    # 1. 加载数据
    products = load_products()
    reviews = load_reviews()
    reviews_by_product = group_reviews_by_product(reviews)

    # 2. 重置集合
    collection = reset_collection("coffee_rag")

    # 3. 逐产品生成向量并存入
    for product in products:
        pid = product["id"]
        product_reviews = reviews_by_product.get(pid, [])
        text = build_product_text(product, product_reviews)
        vector = get_embedding(text)

        collection.add(
            embeddings=[vector],
            documents=[text],
            metadatas=[build_product_metadata(product)],
            ids=[pid],
        )

    print(f"✅ 成功存入 {len(products)} 条咖啡数据（含用户评价）")


if __name__ == "__main__":
    load_and_embed_all()
