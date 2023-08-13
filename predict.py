import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel


import argparse

def predict_ner(sentence):#, model_name):

    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))

    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.strip().split()
    encodded_word_list=[config.TOKENIZER.encode(word,add_special_tokens=False) for word in sentence]

    test_dataset = dataset.EntityDataset(
        texts=[sentence],  
        tags=[[0] * len(sentence)]
    )

    device = torch.device("cuda" )
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)

    prediction= enc_tag.inverse_transform(
                    tag.argmax(2).cpu().numpy().reshape(-1)
                )[:len(tokenized_sentence)][1:-1]       


    def most_common_element(lst):
        return max(set(lst), key=lst.count)

    final_prediction=[]
    j=0
    for encodded_word in encodded_word_list:
      tag=most_common_element(list(prediction[j:j+len(encodded_word)]))
      tag= tag[2:] if len(tag)>1 else tag
      final_prediction.append(tag)
      j = j+len(encodded_word)

    print(f'input text_splited: {sentence}')
    print(f'prediction du model: {final_prediction}')
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Named Entity Recognition using BioBERT model.")
    parser.add_argument("--sentence", required=True, help="Input sentence for NER prediction.")
    args = parser.parse_args()

    predict_ner(args.sentence)
