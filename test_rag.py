from src.rag_utils import RAGModule

def get_knowledge_for_features():
    # 1. 初始化 RAG 模块
    # 注意：请根据你的实际环境修改 index_path 和 corpus_path 的路径
    rag = RAGModule(
        index_path="/workspace/yp/MPO/datasets/rag_index/e5_Flat.index", 
        corpus_path="/workspace/yp/MPO/datasets/FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl",
        model_name="/workspace/yp/MPO/datasets/intfloat_e5-base-v2"
    )

    # 2. 定义从 <Image_Features> 中提取的实体描述
    # features = "Grey head, white throat, orange-rufous belly, grey-brown wings"
    # features="bird,Grey head, black supercilium / mask, white throat, orange-rufous belly, black curved beak, grey-brown wings, black eyes, long tail, perching on tree branch"
    # features="Grey head, White throat, Orange-rufous belly, Black curved beak, Grey-brown wings"
    features=" Grey head, White throat, Orange-rufous belly, Black curved beak, Grey-brown wings,bird"

    # 3. 执行检索
    # top_k 指定返回最相关的外部知识条数
    print(f"正在为特征召回知识: {features}...\n")
    retrieved_knowledge = rag.retrieve(query=features, top_k=10)

    # 4. 输出结果
    if retrieved_knowledge:
        print("--- 召回的外部知识 ---")
        print(retrieved_knowledge)
    else:
        print("未找到相关的外部知识。")

if __name__ == "__main__":
    get_knowledge_for_features()