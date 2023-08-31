import mindspore
from mindnlp.models import BertForSequenceClassification
from mindnlp.transforms.tokenizers import BertTokenizer
import numpy as np
import csv
from mindspore import Tensor


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
param_dict = mindspore.load_checkpoint("checkpoint/bert_qabot_best.ckpt")
mindspore.load_param_into_net(model, param_dict)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_single(question, answer):
    label_map = {0: "错误", 1: "正确"}
    ques = tokenizer.encode(question).ids
    ans = tokenizer.encode(answer).ids
    text_tokenized = np.concatenate((ques, ans[1:]))
    logits = model(Tensor([text_tokenized]))
    predict_label = logits[0].asnumpy().argmax()
    info = f"inputs: '{question} {answer}', predict: '{label_map[predict_label]}'"
    return info

def predict_file(path):
    correct = 0
    total = 0
    with open(path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        next(reader) # Skip the header
        for row in reader:
            question, answer = row
            result = predict_single(question, answer)
            print(result)
            if "正确" in result:
                correct += 1
            total += 1

    score = (correct / total) * 100
    return f"Accuracy: {score:.2f}%"

mode = input("Choose your mode(single/file):").strip().lower()
if mode == "single":
    question = input("Enter your question: ")
    answer = input("Enter your answer: ")
    print(predict_single(question, answer))
elif mode == "file":
    path = input("Enter path to your TSV file: ")
    print(predict_file(path))
else:
    print("Invalid mode chosen.")
