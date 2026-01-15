import logging
import os
from typing import Dict, List, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings
    try:
        from chromadb.errors import ChromaDBError
    except ImportError:
        # ChromaDB 1.4+ 使用 ChromaError 替代 ChromaDBError
        from chromadb.errors import ChromaError as ChromaDBError
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not installed. Install it with: pip install chromadb")

from src.ingestion.base import Document
from src.vector_store.base import VectorStore


try:
    import uuid
except ImportError:
    print("Warning: uuid module not available")


class ChromaVectorStore(VectorStore):
    """Chroma向量数据库实现"""
    
    def __init__(
        self,
        collection_name: str,
        embedding_dimensions: int,
        vector_store_path: str = "./vector_store",
        embedding_function: Optional[callable] = None
    ):
        """初始化Chroma向量数据库
        
        Args:
            collection_name: 集合名称
            embedding_dimensions: 向量维度
            vector_store_path: 向量数据库存储路径
            embedding_function: 嵌入函数（可选）
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not installed. Install it with: pip install chromadb")
        
        super().__init__(
            collection_name=collection_name,
            embedding_dimensions=embedding_dimensions
        )
        
        self.vector_store_path = vector_store_path
        self.embedding_function = embedding_function
        
        # 初始化 logger（在异常处理之前）
        self.logger = logging.getLogger(__name__)
        
        # 确保存储路径存在
        os.makedirs(vector_store_path, exist_ok=True)
        
        try:
            # 初始化Chroma客户端
            self.client = chromadb.PersistentClient(
                path=vector_store_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # ChromaDB 1.4+ 版本中，如果传递 embedding_function，需要使用 ChromaDB 的 EmbeddingFunction 类
            # 如果只是普通函数，传递 None，我们手动处理 embedding
            # 获取或创建集合（不传递 embedding_function，我们手动管理向量）
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,  # 不传递 embedding_function，手动管理向量
                metadata={"embedding_dimensions": embedding_dimensions}
            )
            
            self.logger.info(f"Chroma向量数据库已初始化，集合: {collection_name}")
            
        except ChromaDBError as e:
            self.logger.error(f"Chroma数据库初始化失败: {e}")
            raise RuntimeError(f"Chroma数据库初始化失败: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"向量数据库初始化过程中发生未知错误: {e}")
            raise RuntimeError(f"向量数据库初始化失败: {str(e)}")
    
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
            # 准备数据
            ids = []
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                # 使用文档ID或生成新ID
                doc_id = doc.id or str(uuid.uuid4())
                ids.append(doc_id)
                texts.append(doc.text)
                
                # 清理 metadata，移除所有 None 值（ChromaDB 1.4+ 不允许 None 值）
                cleaned_metadata = {
                    k: v for k, v in doc.metadata.items() 
                    if v is not None
                }
                metadatas.append(cleaned_metadata)
            
            # 添加到Chroma
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            self.logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
            return ids
        
        except ChromaDBError as e:
            self.logger.error(f"添加文档到Chroma失败: {e}")
            raise RuntimeError(f"添加文档失败: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"添加文档过程中发生未知错误: {e}")
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
        
        # 创建文档列表，清理 metadata 中的 None 值
        documents = []
        for i, text in enumerate(texts):
            doc_metadata = metadata[i] if metadata else {}
            # 清理 None 值
            cleaned_metadata = {
                k: v for k, v in doc_metadata.items() 
                if v is not None
            }
            documents.append(Document(text=text, metadata=cleaned_metadata))
        
        return self.add_documents(documents, embeddings)
    
    def search_by_vector(self, query_vector: List[float], top_k: int = 4) -> List[Tuple[Document, float]]:
        """根据向量查询相关文档
        
        Args:
            query_vector: 查询向量
            top_k: 返回的文档数量
            
        Returns:
            List[Tuple[Document, float]]: 文档和相似度得分的列表
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 解析结果
            documents_with_scores = []
            
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    text = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 0.0
                    
                    # 相似度得分（1 - 距离）
                    similarity_score = 1.0 - distance
                    
                    document = Document(text=text, metadata=metadata)
                    documents_with_scores.append((document, similarity_score))
            
            return documents_with_scores
        
        except ChromaDBError as e:
            self.logger.error(f"向量查询失败: {e}")
            raise RuntimeError(f"向量查询失败: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"查询过程中发生未知错误: {e}")
            raise RuntimeError(f"查询失败: {str(e)}")
    
    def search_by_text(self, query_text: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """根据文本查询相关文档
        
        Args:
            query_text: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            List[Tuple[Document, float]]: 文档和相似度得分的列表
        """
        if not self.embedding_function:
            raise RuntimeError("未提供嵌入函数，无法通过文本查询")
        
        try:
            # 生成查询向量
            query_vector = self.embedding_function(query_text)
            
            # 执行向量查询
            return self.search_by_vector(query_vector, top_k=top_k)
        
        except Exception as e:
            self.logger.error(f"文本查询失败: {e}")
            raise RuntimeError(f"文本查询失败: {str(e)}")
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """根据文档ID获取文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            Optional[Document]: 文档，如果不存在则返回None
        """
        try:
            results = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas"]
            )
            
            if results["documents"] and results["documents"][0]:
                return Document(
                    text=results["documents"][0],
                    metadata=results["metadatas"][0] if results["metadatas"] else {},
                    id=document_id
                )
            
            return None
        
        except ChromaDBError as e:
            self.logger.error(f"获取文档失败: {e}")
            raise RuntimeError(f"获取文档失败: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"获取文档过程中发生未知错误: {e}")
            raise RuntimeError(f"获取文档失败: {str(e)}")
    
    def delete_document(self, document_id: str) -> bool:
        """根据文档ID删除文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        try:
            self.collection.delete(ids=[document_id])
            self.logger.info(f"文档已删除: {document_id}")
            return True
        
        except ChromaDBError as e:
            self.logger.error(f"删除文档失败: {e}")
            return False
        
        except Exception as e:
            self.logger.error(f"删除文档过程中发生未知错误: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """删除整个集合
        
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"集合已删除: {self.collection_name}")
            return True
        
        except ChromaDBError as e:
            self.logger.error(f"删除集合失败: {e}")
            return False
        
        except Exception as e:
            self.logger.error(f"删除集合过程中发生未知错误: {e}")
            return False
    
    def get_collection_size(self) -> int:
        """获取集合中文档数量
        
        Returns:
            int: 文档数量
        """
        try:
            return self.collection.count()
        
        except ChromaDBError as e:
            self.logger.error(f"获取集合大小失败: {e}")
            return 0
        
        except Exception as e:
            self.logger.error(f"获取集合大小过程中发生未知错误: {e}")
            return 0
    
    def close(self) -> None:
        """关闭向量数据库连接
        """
        try:
            # Chroma的PersistentClient会自动管理连接，这里主要是做一些清理工作
            self.logger.info(f"关闭Chroma向量数据库连接，集合: {self.collection_name}")
        
        except Exception as e:
            self.logger.error(f"关闭向量数据库连接时发生错误: {e}")


# 工厂函数：创建向量数据库实例
def create_vector_store(
    store_type: str = "chroma",
    collection_name: str = "rag_agent",
    embedding_dimensions: int = 1024,
    vector_store_path: str = "./vector_store",
    embedding_function: Optional[callable] = None,
    persist_directory: Optional[str] = None
) -> VectorStore:
    """创建向量数据库实例
    
    Args:
        store_type: 向量数据库类型（默认 chroma）
        collection_name: 集合名称
        embedding_dimensions: 向量维度
        vector_store_path: 向量数据库存储路径
        embedding_function: 嵌入函数（可选）
        persist_directory: 持久化目录（与vector_store_path相同，为兼容性保留）
        
    Returns:
        VectorStore: 向量数据库实例
    """
    if store_type == "chroma":
        # 使用 persist_directory 或 vector_store_path
        path = persist_directory or vector_store_path
        return ChromaVectorStore(
            collection_name=collection_name,
            embedding_dimensions=embedding_dimensions,
            vector_store_path=path,
            embedding_function=embedding_function
        )
    
    raise ValueError(f"不支持的向量数据库类型: {store_type}")

