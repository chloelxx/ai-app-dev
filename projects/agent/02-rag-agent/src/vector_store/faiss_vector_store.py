import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.ingestion.base import Document
from src.vector_store.base import VectorStore

try:
    import uuid
except ImportError:
    print("Warning: uuid module not available")


class FAISSVectorStore(VectorStore):
    """FAISS向量数据库实现"""
    
    def __init__(
        self,
        collection_name: str,
        embedding_dimensions: int,
        vector_store_path: str = "./vector_store",
        embedding_function: Optional[callable] = None
    ):
        """初始化FAISS向量数据库
        
        Args:
            collection_name: 集合名称
            embedding_dimensions: 向量维度
            vector_store_path: 向量数据库存储路径
            embedding_function: 嵌入函数（可选）
        """
        super().__init__(
            collection_name=collection_name,
            embedding_dimensions=embedding_dimensions
        )
        
        self.vector_store_path = vector_store_path
        self.embedding_function = embedding_function
        self.index_path = os.path.join(vector_store_path, f"{collection_name}.index")
        self.data_path = os.path.join(vector_store_path, f"{collection_name}.pkl")
        
        # 确保存储路径存在
        os.makedirs(vector_store_path, exist_ok=True)
        
        try:
            # 加载或创建索引
            if os.path.exists(self.index_path) and os.path.exists(self.data_path):
                # 加载已有索引和数据
                self.index = faiss.read_index(self.index_path)
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.document_map = data['document_map']  # id -> Document
                    self.vector_map = data['vector_map']  # id -> vector
            else:
                # 创建新索引和数据结构
                self.index = faiss.IndexFlatL2(embedding_dimensions)
                self.document_map = {}
                self.vector_map = {}
            
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"FAISS向量数据库已初始化，集合: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"FAISS数据库初始化失败: {e}")
            raise RuntimeError(f"FAISS数据库初始化失败: {str(e)}")
    
    def _save(self):
        """保存索引和数据到磁盘"""
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, 'wb') as f:
            pickle.dump({
                'document_map': self.document_map,
                'vector_map': self.vector_map
            }, f)
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> List[str]:
        """添加文档和对应的向量到数据库
        
        Args:
            documents: 文档列表
            embeddings: 对应的向量列表
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        if not documents:
            return []
        
        if len(documents) != len(embeddings):
            raise ValueError("文档数量和向量数量不匹配")
        
        try:
            added_ids = []
            new_embeddings = []
            new_ids = []
            
            for i, doc in enumerate(documents):
                # 使用文档ID或生成新ID
                doc_id = doc.id or str(uuid.uuid4())
                
                # 如果文档已存在，跳过或更新
                if doc_id in self.document_map:
                    continue
                    
                # 添加到索引和映射
                new_embeddings.append(embeddings[i])
                new_ids.append(doc_id)
                self.document_map[doc_id] = doc
                self.vector_map[doc_id] = embeddings[i]
                added_ids.append(doc_id)
            
            if new_embeddings:
                # 将新向量添加到FAISS索引
                self.index.add(np.array(new_embeddings, dtype=np.float32))
                # 保存到磁盘
                self._save()
            
            return added_ids
            
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            raise RuntimeError(f"添加文档失败: {str(e)}")
    
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict]] = None) -> List[str]:
        """添加文本和对应的向量到数据库
        
        Args:
            texts: 文本列表
            embeddings: 对应的向量列表
            metadata: 元数据列表
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        if not texts:
            return []
        
        if len(texts) != len(embeddings):
            raise ValueError("文本数量和向量数量不匹配")
        
        if metadata and len(texts) != len(metadata):
            raise ValueError("文本数量和元数据数量不匹配")
        
        # 创建文档对象
        documents = []
        for i, text in enumerate(texts):
            doc_metadata = metadata[i] if metadata else {}
            doc = Document(
                id=None,  # 将在add_documents中生成
                text=text,
                metadata=doc_metadata,
                source="text_input"
            )
            documents.append(doc)
        
        return self.add_documents(documents, embeddings)
    
    def search_by_vector(self, query_vector: List[float], top_k: int = 4) -> List[Tuple[Document, float]]:
        """根据向量查询相关文档
        
        Args:
            query_vector: 查询向量
            top_k: 返回的文档数量
            
        Returns:
            List[Tuple[Document, float]]: 文档和相似度得分的列表
        """
        if not query_vector:
            return []
        
        try:
            # 转换为numpy数组
            query_np = np.array([query_vector], dtype=np.float32)
            
            # 搜索
            distances, indices = self.index.search(query_np, top_k)
            
            # 获取结果
            results = []
            id_list = list(self.document_map.keys())
            
            for i, idx in enumerate(indices[0]):
                if idx < len(id_list):  # 确保索引有效
                    doc_id = id_list[idx]
                    doc = self.document_map[doc_id]
                    score = distances[0][i]
                    results.append((doc, score))
            
            return results
            
        except Exception as e:
            self.logger.error(f"向量搜索失败: {e}")
            raise RuntimeError(f"向量搜索失败: {str(e)}")
    
    def search_by_text(self, query_text: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """根据文本查询相关文档
        
        Args:
            query_text: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            List[Tuple[Document, float]]: 文档和相似度得分的列表
        """
        if not query_text:
            return []
        
        if not self.embedding_function:
            raise ValueError("搜索文本需要嵌入函数")
        
        try:
            # 生成查询向量
            query_vector = self.embedding_function(query_text)
            return self.search_by_vector(query_vector, top_k)
            
        except Exception as e:
            self.logger.error(f"文本搜索失败: {e}")
            raise RuntimeError(f"文本搜索失败: {str(e)}")
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """根据文档ID获取文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            Optional[Document]: 文档，如果不存在则返回None
        """
        return self.document_map.get(document_id)
    
    def delete_document(self, document_id: str) -> bool:
        """根据文档ID删除文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        if document_id not in self.document_map:
            return False
        
        try:
            # 从映射中删除
            del self.document_map[document_id]
            del self.vector_map[document_id]
            
            # 重新构建索引
            self._rebuild_index()
            
            return True
            
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            return False
    
    def _rebuild_index(self):
        """重新构建索引"""
        # 创建新索引
        self.index = faiss.IndexFlatL2(self.embedding_dimensions)
        
        # 添加所有向量
        if self.vector_map:
            vectors = np.array(list(self.vector_map.values()), dtype=np.float32)
            self.index.add(vectors)
        
        # 保存
        self._save()
    
    def delete_collection(self) -> bool:
        """删除整个集合
        
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        try:
            # 清空内存中的数据
            self.index = faiss.IndexFlatL2(self.embedding_dimensions)
            self.document_map.clear()
            self.vector_map.clear()
            
            # 删除磁盘上的文件
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.data_path):
                os.remove(self.data_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"删除集合失败: {e}")
            return False
    
    def get_collection_size(self) -> int:
        """获取集合中文档数量
        
        Returns:
            int: 文档数量
        """
        return len(self.document_map)

def create_vector_store(
    store_type: str = "faiss",
    collection_name: str = "rag_agent",
    embedding_dimensions: int = 1024,
    vector_store_path: str = "./vector_store",
    embedding_function: Optional[callable] = None
) -> VectorStore:
    """创建向量数据库实例
    
    Args:
        store_type: 向量数据库类型
        collection_name: 集合名称
        embedding_dimensions: 向量维度
        vector_store_path: 向量数据库存储路径
        embedding_function: 嵌入函数（可选）
        
    Returns:
        VectorStore: 向量数据库实例
    """
    if store_type == "faiss":
        return FAISSVectorStore(
            collection_name=collection_name,
            embedding_dimensions=embedding_dimensions,
            vector_store_path=vector_store_path,
            embedding_function=embedding_function
        )
    
    raise ValueError(f"不支持的向量数据库类型: {store_type}")