import torch
import torch.nn as nn
from transformers import BertTokenizer

def load_model(path, device):
    class StudentModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, tokenizer):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.pad_token_id)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.25)
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
            self.match_hidden = nn.Linear(hidden_dim * 2, 768)
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            last_hidden = torch.mean(lstm_out, dim=1)
            last_hidden = self.dropout(last_hidden)
            matched_hidden = self.match_hidden(last_hidden)
            logits = self.classifier(last_hidden)
            return logits, matched_hidden
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = StudentModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        num_classes=4,
        tokenizer=tokenizer
    ).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer 