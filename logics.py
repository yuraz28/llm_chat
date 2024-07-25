import numpy as np
from scipy.spatial.distance import cdist
import torch
from model_ml import train_list, label_list, tokenizer_temp, model_temp, device, tokenizer, model
from morze_about_answers import answers


class CustomSentenceTransformer():
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, text):
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings.cpu().numpy()


def get_best(query: str, model, label_list: list = label_list, max_value: int = 10, K: int = 3):
    query_embedding = model.encode([query])
    distances = np.array(cdist(query_embedding, embeddings, "cosine")[0])
    ind = np.argsort(distances, axis=0)
    distances = distances[ind]
    label_list = np.array(label_list)[ind]
    mask = np.isin(distances, distances[distances < max(max_value, min(distances) + 0.01)])

    return {
        'input_text': query,
        'output_texts': label_list[mask][:K],
        'output_distances': distances[mask][:K]
    }


def out_text(dialog: list, isin_bd: bool, tokenizer, model, exemple: list = [], temperature: float = 0.2) -> str:
    text = '<SC6>Ты тестовый бот по имени Morze. Можешь рассказать о команде разработчиков MorzeCrew. Любишь помогать собеседнику. Продолжи диалог:' + '\n'.join(
        dialog[:-1])

    if isin_bd:
        text = text + f'''
        Перефразируй текст: {', '.join(exemple[:3])}
        Клиент: {dialog[-1]}
        Ты: <extra_id_0>'''
    else:
        text = text + f'''
        Ответь на следующий вопрос.
        Клиент: {dialog[-1]}
        Ты: <extra_id_0>'''
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    out_ids = model.generate(input_ids=input_ids.to(device), do_sample=True, temperature=temperature,
                             max_new_tokens=512, top_p=0.85,
                             top_k=2, repetition_penalty=1.2)
    t5_output = tokenizer.decode(out_ids[0][1:])
    t5_output = t5_output.replace('<extra_id_0>', '').strip()
    if '</s>' in t5_output:
        t5_output = t5_output[:t5_output.find('</s>')].strip()

    return t5_output


def pipeline(inp) -> str:
    dct = get_best(inp[-1], search_model, K=1)
    if dct['output_distances'][0] < 0.02:
        exemple = answers[dct['output_texts'][0]]
        return out_text(inp, True, tokenizer, model, exemple=exemple)
    else:
        return out_text(inp, False, tokenizer, model)


search_model = CustomSentenceTransformer(model_temp, tokenizer_temp)
embeddings = search_model.encode(train_list)
