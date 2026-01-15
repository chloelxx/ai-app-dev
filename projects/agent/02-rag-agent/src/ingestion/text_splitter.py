from typing import List, Optional, Sequence

from src.ingestion.base import ChunkResult, Document, TextSplitter


class RecursiveCharacterTextSplitter(TextSplitter):
    """递归字符文本分块器，按分隔符列表递归拆分文本"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        separators: Optional[Sequence[str]] = None,
    ):
        """初始化文本分块器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 文本块之间的重叠字符数
            separators: 用于拆分文本的分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str, **kwargs) -> List[str]:
        """将文本拆分为字符串列表
        
        Args:
            text: 要拆分的文本
            **kwargs: 额外参数
            
        Returns:
            List[str]: 文本块列表
        """
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)
        separators = kwargs.get("separators", self.separators)
        
        def _split_text_with_separator(text: str, separator: str) -> List[str]:
            """使用指定分隔符拆分文本"""
            if separator == "":
                return list(text)
            
            parts = text.split(separator)
            result = []
            current_part = ""
            
            for part in parts:
                if len(current_part) + len(part) + len(separator) <= chunk_size:
                    if current_part:
                        current_part += separator + part
                    else:
                        current_part = part
                else:
                    if current_part:
                        result.append(current_part)
                    current_part = part
            
            if current_part:
                result.append(current_part)
            
            return result
        
        chunks = [text]
        
        for separator in separators:
            temp_chunks = []
            for chunk in chunks:
                if len(chunk) > chunk_size:
                    temp_chunks.extend(_split_text_with_separator(chunk, separator))
                else:
                    temp_chunks.append(chunk)
            chunks = temp_chunks
        
        # 处理重叠
        if chunk_overlap > 0:
            final_chunks = []
            for i, chunk in enumerate(chunks):
                final_chunks.append(chunk)
                if i < len(chunks) - 1:
                    next_chunk = chunks[i + 1]
                    overlap_text = chunk[-chunk_overlap:] + next_chunk[:chunk_overlap]
                    if len(overlap_text) > chunk_overlap:
                        final_chunks[-1] = final_chunks[-1][:-chunk_overlap] + overlap_text
        else:
            final_chunks = chunks
        
        # 过滤空块
        final_chunks = [chunk.strip() for chunk in final_chunks if chunk.strip()]
        
        return final_chunks
    
    def split_document(self, document: Document, **kwargs) -> ChunkResult:
        """将文档拆分为文档块
        
        Args:
            document: 要拆分的文档
            **kwargs: 额外参数
            
        Returns:
            ChunkResult: 分块结果
        """
        chunks = self.split_text(document.text, **kwargs)
        
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            # 创建新的元数据，包含块信息
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "document_type": "chunk",
                "chunk_id": i,
                "chunk_total": len(chunks),
                "chunk_size": len(chunk),
                "parent_id": document.id
            })
            
            chunk_document = Document(
                text=chunk,
                metadata=chunk_metadata,
                id=f"{document.id}_chunk_{i}" if document.id else None
            )
            
            chunk_documents.append(chunk_document)
        
        return ChunkResult(
            chunks=chunk_documents,
            original_document=document,
            chunk_count=len(chunk_documents)
        )
    
    def split_documents(self, documents: List[Document], **kwargs) -> List[ChunkResult]:
        """将多个文档拆分为文档块
        
        Args:
            documents: 要拆分的文档列表
            **kwargs: 额外参数
            
        Returns:
            List[ChunkResult]: 分块结果列表
        """
        results = []
        for document in documents:
            results.append(self.split_document(document, **kwargs))
        return results


class FixedSizeTextSplitter(TextSplitter):
    """固定大小文本分块器，按固定字符数拆分文本"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """初始化文本分块器
        
        Args:
            chunk_size: 每个文本块的字符数
            chunk_overlap: 文本块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str, **kwargs) -> List[str]:
        """将文本拆分为固定大小的字符串列表
        
        Args:
            text: 要拆分的文本
            **kwargs: 额外参数
            
        Returns:
            List[str]: 文本块列表
        """
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)
        
        if chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")
        
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap不能为负数")
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
                
            start = end - chunk_overlap
        
        return chunks
    
    def split_document(self, document: Document, **kwargs) -> ChunkResult:
        """将文档拆分为固定大小的文档块
        
        Args:
            document: 要拆分的文档
            **kwargs: 额外参数
            
        Returns:
            ChunkResult: 分块结果
        """
        chunks = self.split_text(document.text, **kwargs)
        
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "document_type": "chunk",
                "chunk_id": i,
                "chunk_total": len(chunks),
                "chunk_size": len(chunk),
                "parent_id": document.id
            })
            
            chunk_document = Document(
                text=chunk,
                metadata=chunk_metadata,
                id=f"{document.id}_chunk_{i}" if document.id else None
            )
            
            chunk_documents.append(chunk_document)
        
        return ChunkResult(
            chunks=chunk_documents,
            original_document=document,
            chunk_count=len(chunk_documents)
        )
    
    def split_documents(self, documents: List[Document], **kwargs) -> List[ChunkResult]:
        """将多个文档拆分为固定大小的文档块
        
        Args:
            documents: 要拆分的文档列表
            **kwargs: 额外参数
            
        Returns:
            List[ChunkResult]: 分块结果列表
        """
        results = []
        for document in documents:
            results.append(self.split_document(document, **kwargs))
        return results