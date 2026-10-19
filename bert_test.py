from transformers import BertTokenizer, BertModel

# 将 model_path 替换为你下载的BERT模型文件夹的实际路径
model_path = "bert-base-chinese" 

# 加载分词器 (Tokenizer)
tokenizer = BertTokenizer.from_pretrained(model_path)

# 加载基础模型 (BertModel)
model = BertModel.from_pretrained(model_path)

