# src/retriever/hybrid_retriever.py
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.retriever.base import Retriever
from src.ingestion.base import Document
from src.retriever.vector_store_retriever import VectorStoreRetriever
from src.retriever.bm25_retriever import BM25Retriever


def _normalize_scores(scores: List[float]) -> List[float]:
    """将分数归一化到 [0, 1] 区间"""
    if not scores:
        return []
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def _get_doc_id(doc: Document) -> str:
    """获取文档唯一ID（优先用 metadata.id，否则用文本哈希）"""
    if hasattr(doc, 'metadata') and doc.metadata.get('id'):
        return str(doc.metadata['id'])
    return str(hash(doc.text))


class HybridRetriever(Retriever):
    """
    混合检索器：结合向量检索 + BM25 检索
    """
    
    def __init__(
        self,
        vector_retriever: VectorStoreRetriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        """
        初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器实例
            bm25_retriever: BM25检索器实例
            vector_weight: 向量检索权重（建议 0.5～0.7）
            bm25_weight: BM25检索权重（建议 0.3～0.5）
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self._retrieval_count = 0
    
    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """仅返回文档"""
        results = self.retrieve_with_score(query, k, **kwargs)
        return [item["document"] for item in results]
    
    def retrieve_with_score(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """混合检索主逻辑"""
        self._retrieval_count += 1
        
        # 1. 分别检索（取更多结果用于融合）
        vec_results = self.vector_retriever.retrieve_with_score(query, k=k * 2)
        bm25_results = self.bm25_retriever.retrieve_with_score(query, k=k * 2)
        
        # 2. 归一化分数
        vec_scores = _normalize_scores([r["score"] for r in vec_results])
        bm25_scores = _normalize_scores([r["score"] for r in bm25_results])
        
        print("向量检索归一化分数：", vec_scores)
        print("BM25检索归一化分数：", bm25_scores)
        # 3. 构建文档ID到分数的映射
        vec_map = {}
        for i, res in enumerate(vec_results):
            doc_id = _get_doc_id(res["document"])
            vec_map[doc_id] = vec_scores[i]
        
        bm25_map = {}
        for i, res in enumerate(bm25_results):
            doc_id = _get_doc_id(res["document"])
            bm25_map[doc_id] = bm25_scores[i]
        
        # 4. 合并所有文档ID
        all_doc_ids = set(vec_map.keys()) | set(bm25_map.keys())
        
        # 5. 计算混合分数
        hybrid_scores = {}
        for doc_id in all_doc_ids:
            score = (
                self.vector_weight * vec_map.get(doc_id, 0.0) +
                self.bm25_weight * bm25_map.get(doc_id, 0.0)
            )
            hybrid_scores[doc_id] = score
        
        # 6. 排序并取 top-k
        sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # 7. 重建结果（需从原始结果中找回 Document 对象）
        doc_id_to_doc = {}
        for res in vec_results + bm25_results:
            doc_id = _get_doc_id(res["document"])
            if doc_id not in doc_id_to_doc:
                doc_id_to_doc[doc_id] = res["document"]
        
        return [
            {"document": doc_id_to_doc[doc_id], "score": score}
            for doc_id, score in sorted_items
        ]
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        base_stats = super().get_retrieval_stats()
        return {
            **base_stats,
            "retrieval_count": self._retrieval_count,
            "retriever_type": "HybridRetriever",
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "vector_stats": self.vector_retriever.get_retrieval_stats(),
            "bm25_stats": self.bm25_retriever.get_retrieval_stats()
        }