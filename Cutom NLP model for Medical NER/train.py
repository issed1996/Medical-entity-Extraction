import pandas as pd
import numpy as np
import re

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data1(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, pos, tag, enc_pos, enc_tag



def process_line(text):
  text=' '.join(text.split()[1:])
  pattern = r'<category="([^"]+)">([^<]+)</category>'
  matches = [match for match in re.finditer(pattern, text)]

  parts = []
  prev_end = 0

  for match in matches:
      start, end = match.span()
      parts.append(text[prev_end:start])
      parts.append(match.group())
      prev_end = end

  parts.append(text[prev_end:])


  text_list= []
  tags_type_list=[]
  for part in parts:
    
    if "category=" not in part:
      part_split = part.split()
      text_list.extend(part_split)
      tags_type_list.extend(['O']*len(part_split))

    else:
      match = re.search(pattern, part)
      extracted_entity_text = match.group(2)#hereditary breast and ovarian cancer
      extracted_entity_type = match.group(1)
      part_cleaned = part.replace(match.group(0),match.group(2))

      for i, word in enumerate(part_cleaned.split()) :
        if i==0:
          iob_tag="B-"+ extracted_entity_type
        else:
          iob_tag="I-"+ extracted_entity_type

        text_list.append(word)
        tags_type_list.append(iob_tag)

  return  text_list ,tags_type_list


def process_data(data_path):
  texts=[]
  tags=[]
  with open(data_path, 'r') as f:
    lines = f.readlines()
  for line in lines:
    clean_line_splited,clean_line_splited_tags= process_line(line)
    texts.append(clean_line_splited)
    tags.append(clean_line_splited_tags)
  #print(texts)
  #print(tags)

  flat_tags = [tag for subtags in tags for tag in subtags]
  enc_tag = preprocessing.LabelEncoder()
  enc_tag.fit(flat_tags)
  encodded_tags = [enc_tag.transform(sublist) for sublist in tags]

  #print('hahahhaha')
  #print(encodded_tags)


  return texts,encodded_tags,enc_tag #,tags


    


if __name__ == "__main__":
    sentences, tag, enc_tag = process_data(config.TESTING_FILE)

    meta_data = {
        "enc_tag": enc_tag
    }

    joblib.dump(meta_data, "meta.bin")

    num_tag = len(list(enc_tag.classes_))

    #(
     #   train_sentences,
      #  test_sentences,
       # train_pos,
       # test_pos,
       # train_tag,
       # test_tag
    #) = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1)

    train_sentences = sentences
    test_sentences = sentences
    train_tag = tag
    test_tag = tag
    
    train_dataset = dataset.EntityDataset(
        texts=train_sentences, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sentences, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss,test_accuracy = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss} Valid accuracy = {test_accuracy}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
