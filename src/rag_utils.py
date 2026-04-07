import json
import os
from flashrag.config import Config
from flashrag.retriever import DenseRetriever

class RAGModule:
    def __init__(self, index_path, corpus_path, model_name="/workspace/yp/MPO/datasets/intfloat_e5-base-v2", log_file="rag_retrieval_scores.jsonl"):
        """
        基于 FlashRAG 官方源码规范初始化检索器
        增加 log_file 参数用于持久化保存召回得分
        """
        config_dict = {
            'data_dir': './datasets',           
            'dataset_name': 'mpo_rag',          
            'retrieval_method': model_name,     
            'retrieval_model_path': model_name, 
            'index_path': index_path,      
            'corpus_path': corpus_path,    
            'retrieval_topk': 3,
            'disable_save': True                
        }
        self.config = Config(config_dict=config_dict)
        self.retriever = DenseRetriever(self.config)
        self.log_file = log_file # 设置保存得分的文件路径

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        检索强相关的外部辅助知识，并提取、记录召回相关性得分
        """
        # 核心修改：利用官方 API 的 return_score=True 直接返回 (结果列表, 得分列表)
        results, scores = self.retriever.search(query, num=top_k, return_score=True)
        
        if not results:
            print("[RAG Score] 检索失败，未找到相关文档。")
            return ""
            
        retrieved_docs = [doc['contents'] for doc in results if 'contents' in doc]
        
        # 计算平均得分
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # 1. 终端打印输出
        print(f"[RAG Score] Top-{top_k} 得分: {[round(s, 4) for s in scores]} | 平均分: {avg_score:.4f}")
        
        # 2. 记录保存到本地文件 (JSON Lines 格式，方便后续用 Pandas 等工具分析)
        log_data = {
            "query": query,
            "top_k": top_k,
            "scores": scores,
            "avg_score": avg_score,
            "retrieved_docs_preview": [doc[:50] + "..." for doc in retrieved_docs] # 截断保存部分内容
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[RAG Score] 得分日志写入失败: {e}")
        
        return "\n\n".join(retrieved_docs)