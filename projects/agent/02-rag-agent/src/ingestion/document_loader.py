import os
from typing import List, Optional

from src.ingestion.base import Document, DocumentLoader


try:
    from PyPDF2 import PdfReader
except ImportError:
    print("Warning: PyPDF2 not installed. PDF support will be limited.")


class SimpleDocumentLoader(DocumentLoader):
    """简单文档加载器，支持txt、md和pdf格式"""
    
    def __init__(self, supported_extensions: Optional[List[str]] = None):
        """初始化文档加载器
        
        Args:
            supported_extensions: 支持的文件扩展名列表
        """
        self.supported_extensions = supported_extensions or [".txt", ".md", ".pdf"]
    
    def load(self, file_path: str, **kwargs) -> Document:
        """加载单个文档
        
        Args:
            file_path: 文件路径
            **kwargs: 额外参数
            
        Returns:
            Document: 加载的文档
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {file_extension}")
        
        text = ""
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_extension": file_extension,
            "document_type": "original"
        }
        
        # 根据文件类型加载内容
        if file_extension in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_extension == ".pdf":
            text = self._load_pdf(file_path, metadata)
        
        return Document(
            text=text,
            metadata=metadata,
            id=kwargs.get("id")
        )
    
    def load_directory(self, directory_path: str, **kwargs) -> List[Document]:
        """加载目录中的所有文档
        
        Args:
            directory_path: 目录路径
            **kwargs: 额外参数
            
        Returns:
            List[Document]: 加载的文档列表
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        documents = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()
                
                if file_extension in self.supported_extensions:
                    try:
                        document = self.load(file_path, **kwargs)
                        documents.append(document)
                    except Exception as e:
                        print(f"加载文件失败 {file_path}: {e}")
        
        return documents
    
    def _load_pdf(self, file_path: str, metadata: dict) -> str:
        """加载PDF文件
        
        Args:
            file_path: PDF文件路径
            metadata: 文档元数据
            
        Returns:
            str: PDF文件的文本内容
        """
        try:
            reader = PdfReader(file_path)
            metadata["total_pages"] = len(reader.pages)
            
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n"
                    text += page_text
            
            return text
        except Exception as e:
            raise RuntimeError(f"PDF加载失败: {e}")