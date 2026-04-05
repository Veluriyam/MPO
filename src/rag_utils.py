from flashrag.config import Config
from flashrag.retriever import DenseRetriever

class RAGModule:
    def __init__(self, index_path, corpus_path, model_name="/workspace/yp/MPO/datasets/intfloat_e5-base-v2"):
        """
        初始化 FlashRAG 检索器
        """
        config_dict = {
            'retrieval_method': 'dense',
            'index_path': index_path,      
            'corpus_path': corpus_path,    
            'retriever_model_name_or_path': model_name
        }
        self.config = Config(config_dict=config_dict)
        self.retriever = DenseRetriever(self.config)

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        检索强相关的外部辅助知识
        """
        results = self.retriever.search([query], num=top_k)
        if not results or not results[0]:
            return ""
            
        retrieved_docs = [doc['contents'] for doc in results[0]]
        return "\n\n".join(retrieved_docs)