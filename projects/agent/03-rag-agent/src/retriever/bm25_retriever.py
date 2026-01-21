# src/retriever/bm25_retriever.py
import jieba
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from src.retriever.base import Retriever
from src.ingestion.base import Document


class BM25Retriever(Retriever):
    """
    基于 BM25 的关键词检索器
    """
    
    def __init__(self, documents: List[Document], language: str = "zh"):
        """
        初始化 BM25 检索器
        
        Args:
            documents: 所有文档列表（用于构建索引）
            language: 语言类型 ("zh" 中文 / "en" 英文)
        """
        self.documents = documents
        self.language = language
        self._retrieval_count = 0
        
        # 预处理：提取文本并分词
        tokenized_docs = []
        self.doc_texts = []
        
        for doc in documents:
            text = doc.text
            self.doc_texts.append(text)
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)
        
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """分词函数"""
        if not text.strip():
            return []
        if self.language == "zh":
            return [word for word in jieba.cut(text) if word.strip() and not word.isspace()]
        else:
            return text.lower().split()
    
    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """仅返回文档（不带分数）"""
        results = self.retrieve_with_score(query, k, **kwargs)
        return [item["document"] for item in results]
    
    def retrieve_with_score(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """返回文档 + BM25 分数"""
        self._retrieval_count += 1
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k 索引（按分数降序）
        top_k_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx])  # 转为 float 兼容 JSON
            })
        
        return results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        base_stats = super().get_retrieval_stats()
        return {
            **base_stats,
            "retrieval_count": self._retrieval_count,
            "retriever_type": "BM25Retriever",
            "document_count": len(self.documents),
            "language": self.language
        }