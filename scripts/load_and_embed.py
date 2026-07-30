# 1. 安装依赖
# pip install dashscope chromadb

import os
import json
import dashscope
from dashscope import TextEmbedding
import chromadb
from chromadb.api.types import EmbeddingFunction

# 2. 配置你的API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

# 3. 定义一个函数，把文本转成向量（调用Qwen embedding）
def get_embedding(text):
    resp = TextEmbedding.call(
        model='qwen3.7-text-embedding',  # Qwen的embedding模型
        input=text
    )
    return resp.output['embeddings'][0]['embedding']

# 4. 初始化ChromaDB（持久化模式，数据存到项目根目录的 data/chroma_db）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
chroma_path = os.path.join(data_dir, "chroma_db")
client = chromadb.PersistentClient(path=chroma_path)
# 如果集合已存在，先清空旧数据再重新写入（避免重复运行冲突）
try:
    client.delete_collection(name="coffee_rag")
    print("旧集合已删除，将重新创建")
except Exception:
    pass
collection = client.create_collection(name="coffee_rag")

# 5. 读取产品数据和评价数据
json_path = os.path.join(data_dir, 'coffee_products.json')
reviews_path = os.path.join(data_dir, 'reviews.json')

with open(json_path, 'r', encoding='utf-8') as f:
    products = json.load(f)

# 读取评价，按 product_id 分组
with open(reviews_path, 'r', encoding='utf-8') as f:
    reviews = json.load(f)

reviews_by_product = {}
for review in reviews:
    pid = review['product_id']
    if pid not in reviews_by_product:
        reviews_by_product[pid] = []
    reviews_by_product[pid].append(review)

# 6. 为每个产品拼接完整信息（含用户评价），生成向量并存入
for product in products:
    # 拼接产品信息
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
        f"详细描述：{product['description']}"
    ]

    # 追加该产品的用户评价
    if product['id'] in reviews_by_product:
        text_parts.append("用户评价：")
        for r in reviews_by_product[product['id']]:
            text_parts.append(f"  - {r['user']}（{r['rating']}星）：{r['comment']}")

    text_to_embed = "。".join(text_parts)

    # 生成向量
    vector = get_embedding(text_to_embed)
    # 存入ChromaDB（带上元数据）
    collection.add(
        embeddings=[vector],
        documents=[text_to_embed],
        metadatas=[{
            "id": product['id'],
            "name": product['name'],
            "roast_level": product['roast_level'],
            "acidity_level": product['acidity_level'],
            "price": product['price'],
            "category": product['category']
        }],
        ids=[product['id']]
    )

print(f"✅ 成功存入 {len(products)} 条咖啡数据（含用户评价）")
