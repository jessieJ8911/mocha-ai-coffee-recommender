"""
向量生成模块
- 配置 DashScope API Key
- 调用 Qwen embedding 模型生成文本向量
- 提供余弦相似度计算工具
"""

import os
import numpy as np
import dashscope
from dashscope import TextEmbedding

# 配置 API Key（从环境变量读取）
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def get_embedding(text: str) -> list[float]:
    """将文本转为向量（调用 Qwen embedding 模型）

    Args:
        text: 待向量化的文本

    Returns:
        浮点数列表形式的向量
    """
    resp = TextEmbedding.call(
        model='qwen3.7-text-embedding',
        input=text
    )
    return resp.output['embeddings'][0]['embedding']


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """批量生成向量（一次 API 调用处理多条文本）

    Args:
        texts: 文本列表

    Returns:
        向量列表
    """
    resp = TextEmbedding.call(
        model='qwen3.7-text-embedding',
        input=texts
    )
    return [item['embedding'] for item in resp.output['embeddings']]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度

    Args:
        vec_a: 向量 A
        vec_b: 向量 B

    Returns:
        余弦相似度，范围 [-1, 1]，越接近 1 越相似
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
