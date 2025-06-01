import gradio as gr
import importlib
import time
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATHS = {
    "Bi-LSTM (distilled)": "demo/models/best_student_model_weights_only.pth",
    "BERT-based model": "demo/models/bert_classifier_weights_only.pth",
    "LSTM": "demo/models/lstm.pth",
    "Bi-LSTM": "demo/models/bilstm.pth",
    "Random Forrest": "demo/models/random_forest_model.pkl",
    "XGBoost": "demo/models/xgb.pkl",
    "Stochastic Gradient Descent": "demo/models/sgdc.pkl"
}

LOADED_MODELS = {}
TFIDF_VECTORIZER = None
BERT_TOKENIZER = None

def get_bert_tokenizer():
    global BERT_TOKENIZER
    if BERT_TOKENIZER is None:
        BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
    return BERT_TOKENIZER

def load_tfidf():
    global TFIDF_VECTORIZER
    if TFIDF_VECTORIZER is None:
        import pickle
        try:
            with open("demo/utils/tfidf_vectorizer.pkl", "rb") as f:
                TFIDF_VECTORIZER = pickle.load(f)
        except FileNotFoundError:
            print("Warning: tfidf_vectorizer.pkl not found. Please ensure the vectorizer file exists.")
            return None
    return TFIDF_VECTORIZER

def dynamic_load_model(model_name, path):
    if model_name == "LSTM":
        module = importlib.import_module(f"utils.load_model_lstm")
        model, _ = module.load_model(path, DEVICE)
        return model, get_bert_tokenizer()
    elif model_name == "Bi-LSTM":
        module = importlib.import_module(f"utils.load_model_bilstm")
        model, _ = module.load_model(path, DEVICE)
        return model, get_bert_tokenizer()
    elif model_name == "BERT-based model":
        module = importlib.import_module(f"utils.load_model_bert_based_model")
        model, _ = module.load_model(path, DEVICE)
        return model, get_bert_tokenizer()
    elif model_name == "Bi-LSTM (distilled)":
        module = importlib.import_module(f"utils.load_model_student")
        model, _ = module.load_model(path, DEVICE)
        return model, get_bert_tokenizer()
    elif model_name in ["Random Forrest", "XGBoost", "Stochastic Gradient Descent"]:
        import joblib
        model = joblib.load(path)
        return model, load_tfidf()

def gradio_predict(text, model_name):
    start_time = time.time()
    if model_name not in LOADED_MODELS:
        model, tokenizer = dynamic_load_model(model_name, MODEL_PATHS[model_name])
        LOADED_MODELS[model_name] = (model, tokenizer)
    else:
        model, tokenizer = LOADED_MODELS[model_name]

    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    if model_name in ["Bi-LSTM (distilled)", "BERT-based model"]:
        import torch.nn.functional as F
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        with torch.no_grad():
            if model_name == "BERT-based model":
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits, _ = model(input_ids)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        result = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
    
    elif model_name in ["LSTM", "Bi-LSTM"]:
        import torch.nn.functional as F
        # Pre-tokenize and move to device in one step
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt'
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = model(encoding['input_ids'])
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        result = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
    
    elif model_name in ["Random Forrest", "XGBoost", "Stochastic Gradient Descent"]:
        # Transform text using TF-IDF
        features = tokenizer.transform([text])
        if model_name == "Stochastic Gradient Descent":
            # SGD classifier outputs decision function values
            scores = model.decision_function(features)[0]
            probs = np.exp(scores) / np.sum(np.exp(scores))
        else:
            probs = model.predict_proba(features)[0]
        result = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
    
    else:
        result = {class_name: 0.0 for class_name in class_names}
    
    elapsed = time.time() - start_time
    return result, f"{elapsed:.4f} sec"

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Textbox(lines=4, label="Enter text", placeholder="Type or paste your news/article here..."),
        gr.Dropdown(choices=list(MODEL_PATHS.keys()), value="Bi-LSTM (distilled)", label="Select model")
    ],
    outputs=[
        gr.Label(num_top_classes=4, label="Soft label (class probabilities)"),
        gr.Textbox(label="Prediction time")
    ],
    title="Text Classification",
    description="<b>Text Classification Demo</b><br>Select a model and enter text to see class probabilities and prediction time.<br>Supports multiple models including BERT, Student, LSTM, BiLSTM, Random Forest, XGBoost, and SGD.",
    allow_flagging="never",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {background: #23272f; color: #e6e6e6;}
    .gr-button {font-size: 18px; background: #6ec1e4 !important; color: #23272f !important;}
    .gr-button:active, .gr-button:focus {background: #4fa3c7 !important;}
    .gr-textbox textarea {font-size: 16px; background: #2c313a; color: #e6e6e6;}
    .gr-label {font-size: 18px;}
    .gr-panel, .gr-box, .gr-block {background: #2c313a !important;}
    .gr-input, .gr-dropdown {background: #2c313a !important; color: #e6e6e6 !important;}
    h1, .gr-title {font-size: 2.8rem !important; font-weight: 800 !important;}
    .gr-description, .gr-markdown {font-size: 1.35rem !important;}
    """
)

iface.launch(share=False) 