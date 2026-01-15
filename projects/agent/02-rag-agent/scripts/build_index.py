#!/usr/bin/env python3
"""
知识库索引构建脚本

此脚本用于：
1. 从指定目录读取文档
2. 解析文档并进行分块
3. 生成文本块的Embedding
4. 将Embedding存储到向量数据库中

使用方法：
python scripts/build_index.py

可选参数：
--document-dir: 指定文档目录路径
--collection-name: 指定向量数据库集合名称
--chunk-size: 指定文档分块大小
--chunk-overlap: 指定文档分块重叠大小
--rebuild: 是否重建索引（删除现有集合）
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_settings
from src.ingestion.document_loader import SimpleDocumentLoader
from src.ingestion.text_splitter import RecursiveCharacterTextSplitter
from src.embeddings.openai_embeddings import create_embedding_client
from src.vector_store.faiss_vector_store import create_vector_store

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="知识库索引构建脚本")
    parser.add_argument(
        "--document-dir",
        type=str,
        default=None,
        help="指定文档目录路径"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="指定向量数据库集合名称"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="指定文档分块大小"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="指定文档分块重叠大小"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="是否重建索引（删除现有集合）"
    )
    
    return parser.parse_args()

def build_index(args):
    """
    构建知识库索引
    """
    logger.info("开始构建知识库索引...")
    
    # 加载配置
    settings = get_settings()
    
    # 使用命令行参数覆盖配置（如果提供）
    document_dir = args.document_dir or settings.document_dir
    collection_name = args.collection_name or settings.vector_store_collection_name
    chunk_size = args.chunk_size or settings.rag_chunk_size
    chunk_overlap = args.chunk_overlap or settings.rag_chunk_overlap
    
    # 验证文档目录
    if not os.path.exists(document_dir):
        logger.error(f"文档目录不存在: {document_dir}")
        return False
    
    logger.info(f"使用的配置:")
    logger.info(f"  文档目录: {document_dir}")
    logger.info(f"  集合名称: {collection_name}")
    logger.info(f"  分块大小: {chunk_size}")
    logger.info(f"  分块重叠: {chunk_overlap}")
    logger.info(f"  向量数据库路径: {settings.vector_store_path}")
    logger.info(f"  Embedding模型: {settings.embedding_model}")
    logger.info(f"  是否重建索引: {args.rebuild}")
    
    try:
        # 1. 初始化文档加载器
        document_loader = SimpleDocumentLoader()
        
        # 2. 加载文档
        logger.info(f"正在从 {document_dir} 加载文档...")
        documents = document_loader.load_directory(document_dir)
        logger.info(f"成功加载 {len(documents)} 个文档")
        
        if not documents:
            logger.warning("未找到任何文档")
            return True
        
        # 3. 初始化文本分块器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 4. 分块处理
        logger.info(f"正在对文档进行分块处理...")
        all_chunks = []
        for doc in tqdm(documents, desc="分块处理"):
            chunks = text_splitter.split_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"分块完成，共生成 {len(all_chunks)} 个文本块")
        
        if not all_chunks:
            logger.warning("未生成任何文本块")
            return True
        
        # 5. 初始化Embedding客户端
        logger.info("正在初始化Embedding客户端...")
        embedding_client = create_embedding_client(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.openai_base_url
        )
        
        # 6. 初始化向量数据库
        logger.info("正在初始化向量数据库...")
        vector_store = create_vector_store(
            persist_directory=settings.vector_store_path,
            embedding_client=embedding_client,
            collection_name=collection_name
        )
        
        # 7. 如果需要重建索引，删除现有集合
        if args.rebuild:
            logger.info(f"正在删除现有集合: {collection_name}")
            vector_store.delete_collection()
            # 重新创建集合
            vector_store = create_vector_store(
                persist_directory=settings.vector_store_path,
                embedding_client=embedding_client,
                collection_name=collection_name
            )
        
        # 8. 向向量数据库添加文本块
        logger.info("正在向向量数据库添加文本块...")
        vector_store.add_documents(all_chunks)
        
        # 9. 持久化向量数据库
        vector_store.persist()
        
        # 10. 获取统计信息
        stats = vector_store.get_stats()
        logger.info(f"索引构建完成")
        logger.info(f"  集合名称: {stats.get('collection_name')}")
        logger.info(f"  文档数量: {stats.get('document_count')}")
        logger.info(f"  向量数量: {stats.get('vector_count')}")
        
        return True
        
    except Exception as e:
        logger.error(f"构建索引时发生错误: {e}", exc_info=True)
        return False


def main():
    """
    主函数
    """
    args = parse_arguments()
    success = build_index(args)
    
    if success:
        logger.info("索引构建成功")
        return 0
    else:
        logger.error("索引构建失败")
        return 1


if __name__ == "__main__":
    exit(main())