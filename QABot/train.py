import os
import tqdm
import mindspore
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn, context

from mindnlp.transforms import PadTransform
from mindnlp.transforms.tokenizers import BertTokenizer

from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy

import csv

class Loader:
    
    def __init__(self, path):
        self.path = path
        self._data = []  # This will store dictionaries
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            next(spamreader, None)  # skip the headers
            for row in spamreader:
                res = {}
                res['question'] = str(row[1])
                res['answer']=str(row[5])
                res['label'] = int(row[6])
                self._data.append(res)

    def __getitem__(self, index):
        return self._data[index]['label'], self._data[index]['question'],self._data[index]['answer']

    def __len__(self):
        return len(self._data)

train_file = Loader('ours/WikiQA-train.tsv')
valid_file = Loader('ours/WikiQA-dev.tsv')
test_file = Loader('ours/WikiQA-test.tsv')

import numpy as np

def process_dataset(source, tokenizer, pad_value, max_seq_len=64, batch_size=32, shuffle=True):
    column_names = ["label", "question",'answer']
    rename_columns = ["label", "input_ids"]
    
    def concat_columns(data1, data2):
        return np.concatenate((data1, data2[1:]), axis=0)  # Skip the first element of data2

    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    pad_op = PadTransform(max_seq_len, pad_value=pad_value)
    type_cast_op = transforms.TypeCast(mindspore.int32)
    
    # map dataset
    dataset = dataset.map(tokenizer, input_columns="question")
    dataset = dataset.map(tokenizer, input_columns="answer")
    dataset = dataset.map(operations=[type_cast_op], input_columns="label")

    # Concatenate question and answer columns and then pad the result
    dataset = dataset.map(operations=concat_columns, input_columns=["question", "answer"], output_columns=["input_ids"], column_order=["label", "input_ids"])
    dataset = dataset.map(operations=pad_op, input_columns="input_ids")  # Apply padding

    # batch dataset
    dataset = dataset.batch(batch_size)

    return dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pad_value = tokenizer.token_to_id('[PAD]')
dataset_train = process_dataset(train_file, tokenizer, pad_value)
dataset_val = process_dataset(valid_file, tokenizer, pad_value)
dataset_test = process_dataset(test_file, tokenizer, pad_value, shuffle=False)

from mindnlp.models import BertForSequenceClassification
from mindnlp._legacy.amp import auto_mixed_precision

# set bert config and define parameters for training
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = auto_mixed_precision(model, 'O1')

loss = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=2e-5)

metric = Accuracy()

# define callbacks to save checkpoints
ckpoint_cb = CheckpointCallback(save_path='checkpoint', ckpt_name='bert_qabot', epochs=1, keep_checkpoint_max=2)
best_model_cb = BestModelCallback(save_path='checkpoint', ckpt_name='bert_qabot_best', auto_load=True)

trainer = Trainer(network=model, train_dataset=dataset_train,
                  eval_dataset=dataset_val, metrics=metric,
                  epochs=5, loss_fn=loss, optimizer=optimizer, callbacks=[ckpoint_cb, best_model_cb],
                  jit=True)

# start training
trainer.run('label')

evaluator = Evaluator(network=model, eval_dataset=dataset_test, metrics=metric)
evaluator.run(tgt_columns="label")