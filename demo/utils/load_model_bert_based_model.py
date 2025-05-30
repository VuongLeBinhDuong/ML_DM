import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

def load_model(path, device):
    class BertClassifier(nn.Module):
        def __init__(self, num_classes):
            super(BertClassifier, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert.gradient_checkpointing_enable()
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs[1]  # [CLS] token representation
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits
    # Sửa số class nếu cần
    num_classes = 4
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertClassifier(num_classes=num_classes).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer 