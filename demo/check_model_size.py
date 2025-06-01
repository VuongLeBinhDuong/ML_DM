import torch
from utils.model_size import print_model_info
from utils.load_model_bert_based_model import load_model as load_bert_model
from utils.load_model_student import load_model as load_student_model
from utils.load_model_lstm import load_model as load_lstm_model
from utils.load_model_bilstm import load_model as load_bilstm_model

def main():
    device = torch.device('cpu')
    
    # Load and check BERT model
    print("\nChecking BERT model...")
    bert_model, _ = load_bert_model("demo/models/bert_classifier_weights_only.pth", device)
    print_model_info(bert_model, "BERT Model")

    # Load and check Student model
    print("\nChecking Student model...")
    student_model, _ = load_student_model("demo/models/best_student_model_weights_only.pth", device)
    print_model_info(student_model, "Student Model")

    # Load and check LSTM model
    print("\nChecking LSTM model...")
    lstm_model, _ = load_lstm_model("demo/models/lstm.pth", device)
    print_model_info(lstm_model, "LSTM Model")

    # Load and check BiLSTM model
    print("\nChecking BiLSTM model...")
    bilstm_model, _ = load_bilstm_model("demo/models/bilstm.pth", device)
    print_model_info(bilstm_model, "BiLSTM Model")

if __name__ == "__main__":
    main() 