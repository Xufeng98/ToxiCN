import torch
from src.Models import Bert_Layer, TwoLayerFFNNLayer
from transformers import AutoTokenizer
from src.datasets import *
from src.Models import *
from model.Config_base import Config_base


# dataset = 'TOC'  # 数据集
dataset = "ToxiCN"
# 指定模型的完整路径
model_path = "/data/coding/ToxinCN-main/ToxiCN_ex/hfl_chinese-roberta-wwm-ext"
# 指定模型的名称
model_name = "hfl_chinese-roberta-wwm-ext"
# 创建Config_base对象
config = Config_base(model_path, model_name, dataset)
print(config.model_name)

# 初始化tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
bert_layer = Bert_Layer(config).to(config.device)
two_layer_ffnn = TwoLayerFFNNLayer(config).to(config.device)

# 加载模型权重，这里需要您提供具体的路径
checkpoint_path = '/data/coding/ToxinCN-main/ToxiCN_ex/ToxiCN/saved_dict/ckp-hfl_chinese-roberta-wwm-ext-NN_ML-80_D-0.5_B-32_E-5_Lr-1e-05_aplha-0.5-BEST.tar'  # 请替换为实际的检查点文件路径
checkpoint = torch.load(checkpoint_path)
bert_layer.load_state_dict(checkpoint['embed_model_state_dict'])
two_layer_ffnn.load_state_dict(checkpoint['model_state_dict'])

# 将模型设置为评估模式
bert_layer.eval()
two_layer_ffnn.eval()

# 创建数据集实例并执行预处理
# 这里假设您想要预测的数据集路径为'path_to_pred_data.csv'
pred_data_path = '/data/coding/ToxinCN-main/ToxiCN_ex/ToxiCN/data/test.json'  # 请替换为实际的预测数据集路径
dataset_class = Datasets(config, pred_data_path, add_special_tokens=True, not_test=False)
dataset_class.preprocess_data()

# 创建DataLoader用于获取预测数据
dataloader = torch.utils.data.DataLoader(dataset_class, batch_size=config.batch_size, shuffle=False)

# 预测函数
def predict(dataloader, bert_layer, two_layer_ffnn):
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            # print(f"Original batch: {batch}")
            args = to_tensor(batch) # 转换为模型需要的输入格式
            att_input, pooled_emb = bert_layer(**args)  # 获取BERT层的输出
            logit = two_layer_ffnn(att_input, pooled_emb)  # 通过分类器进行预测
            pred = torch.argmax(logit, dim=-1).cpu().numpy()  # 获取预测结果
            predictions.extend(pred)
    return predictions

# 执行预测并获取结果
predictions = predict(dataloader, bert_layer, two_layer_ffnn)

# 假设原始数据框df已经加载了预测数据集'path_to_pred_data.csv'
import pandas as pd
df = pd.read_csv(pred_data_path)

# 将预测结果添加到数据框中
df['predictions'] = predictions

# 保存预测结果到CSV
predictions_output_path = 'path_to_save_predictions.csv'  # 请替换为希望保存的预测结果路径
df.to_csv(predictions_output_path, index=False)
print(f"Predictions have been saved to {predictions_output_path}")