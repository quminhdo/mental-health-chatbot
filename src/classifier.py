import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class Model(nn.Module):
    def __init__(self,bert):
        super(Model, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
    def forward(self, inp, labels=None):
        output = self.bert(inp['input_ids'], attention_mask=inp['attention_mask'])
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        return output

class StressClassifier:

    def __init__(self, pretrained_model, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model(AutoModel.from_pretrained(pretrained_model))
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, text):
        encoded_inputs = self.tokenizer(text,
                                max_length=512,
                                padding="max_length",
                                truncation=True)
        inputs = {}
        inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])
        inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])

        out = self.model(inputs)
        return out[0][0] > 0.5


