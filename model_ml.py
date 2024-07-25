import pandas as pd
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sys import platform

device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('data/trainDataSet_morze_about_all_intents.csv', sep=',')


train_list = df.text.to_list()
label_list = df.label.to_list()


tokenizer_temp = AutoTokenizer.from_pretrained(
    "yuraz28/bert-multilingual-passage-reranking-msmarco-onnx-fe-optimized-fp16")
model_temp = ORTModelForFeatureExtraction.from_pretrained(
    "yuraz28/bert-multilingual-passage-reranking-msmarco-onnx-fe-optimized-fp16", use_io_binding=True).to(device)
tokenizer = AutoTokenizer.from_pretrained('morzecrew/FRED-T5-RefinedPersonaChat')
model = T5ForConditionalGeneration.from_pretrained('morzecrew/FRED-T5-RefinedPersonaChat').to(device)

if platform == "linux" or platform == "linux2" or platform == "darwin":
    model.compile()
