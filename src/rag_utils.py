from flashrag.config import Config
from flashrag.retriever import DenseRetriever

class RAGModule:
    def __init__(self, index_path, corpus_path, model_name="/workspace/yp/MPO/datasets/intfloat_e5-base-v2"):
        """
        基于 FlashRAG 官方源码规范初始化检索器
        """
        config_dict = {
            'data_dir': './datasets',           # 必须提供，Config 初始化时会用到
            'dataset_name': 'mpo_rag',          # 必须提供
            'retrieval_method': model_name,     # FlashRAG 通常将模型名称作为检索方法名
            'retrieval_model_path': model_name, # 本地模型的绝对路径
            'index_path': index_path,      
            'corpus_path': corpus_path,    
            'retrieval_topk': 3,
            'disable_save': True                # 关键：禁止 FlashRAG 在后台自动创建无关的 save 日志文件夹
        }
        self.config = Config(config_dict=config_dict)
        self.retriever = DenseRetriever(self.config)

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        检索强相关的外部辅助知识
        """
        # 官方 search API 接受单个 str，返回 List[Dict]
        results = self.retriever.search(query, num=top_k)
        
        if not results:
            return ""
            
        # 官方返回格式为 [{'id': '...', 'title': '...', 'text': '...', 'contents': '...'}, ...]
        retrieved_docs = [doc['contents'] for doc in results if 'contents' in doc]
        
        return "\n\n".join(retrieved_docs)