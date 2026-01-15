from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Document:
    """文档类，表示一个完整的文档或文档块"""
    text: str
    metadata: Dict[str, any] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class ChunkResult:
    """分块结果类"""
    chunks: List[Document]
    original_document: Optional[Document] = None
    chunk_count: int = 0


class DocumentLoader:
    """文档加载器基础接口"""
    
    def load(self, file_path: str, **kwargs) -> Document:
        """加载单个文档
        
        Args:
            file_path: 文件路径
            **kwargs: 额外参数
            
        Returns:
            Document: 加载的文档
        """
        raise NotImplementedError
    
    def load_directory(self, directory_path: str, **kwargs) -> List[Document]:
        """加载目录中的所有文档
        
        Args:
            directory_path: 目录路径
            **kwargs: 额外参数
            
        Returns:
            List[Document]: 加载的文档列表
        """
        raise NotImplementedError


class TextSplitter:
    """文本分块器基础接口"""
    
    def split_text(self, text: str, **kwargs) -> List[str]:
        """将文本拆分为字符串列表
        
        Args:
            text: 要拆分的文本
            **kwargs: 额外参数
            
        Returns:
            List[str]: 文本块列表
        """
        raise NotImplementedError
    
    def split_document(self, document: Document, **kwargs) -> ChunkResult:
        """将文档拆分为文档块
        
        Args:
            document: 要拆分的文档
            **kwargs: 额外参数
            
        Returns:
            ChunkResult: 分块结果
        """
        raise NotImplementedError
    
    def split_documents(self, documents: List[Document], **kwargs) -> List[ChunkResult]:
        """将多个文档拆分为文档块
        
        Args:
            documents: 要拆分的文档列表
            **kwargs: 额外参数
            
        Returns:
            List[ChunkResult]: 分块结果列表
        """
        raise NotImplementedError