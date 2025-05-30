import gradio as gr
import importlib
import time
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATHS = {
    "student": "demo/models/best_student_model_weights_only.pth",
    "bert_based_model": "demo/models/bert_classifier_weights_only.pth"
    # "xgb": "./models/xgb_model.pkl"
    # Add more models here
}

LOADED_MODELS = {}

def dynamic_load_model(model_name, path):
    module = importlib.import_module(f"utils.load_model_{model_name}")
    return module.load_model(path, DEVICE)

def gradio_predict(text, model_name):
    start_time = time.time()
    if model_name not in LOADED_MODELS:
        model, tokenizer = dynamic_load_model(model_name, MODEL_PATHS[model_name])
        LOADED_MODELS[model_name] = (model, tokenizer)
    else:
        model, tokenizer = LOADED_MODELS[model_name]

    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    if model_name == "student":
        import torch.nn.functional as F
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        with torch.no_grad():
            logits, _ = model(input_ids)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        result = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
    elif model_name == "bert_based_model":
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
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        result = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
    elif model_name == "xgb":
        features = [[len(text)]]
        probs = model.predict_proba(features)[0]
        result = {f'class_{i}': float(prob) for i, prob in enumerate(probs)}
    else:
        result = {f'class_{i}': 0.0 for i in range(4)}
    elapsed = time.time() - start_time
    return result, f"{elapsed:.4f} sec"

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Textbox(lines=4, label="Enter text"),
        gr.Dropdown(choices=list(MODEL_PATHS.keys()), value="student", label="Select model")
    ],
    outputs=[
        gr.Label(num_top_classes=4, label="Soft label (class probabilities)"),
        gr.Textbox(label="Prediction time")
    ],
    title="Text Classification - Multi Model",
    description="Select a model and enter text to see class probabilities and prediction time.",
    allow_flagging="never"
)

iface.launch(share=True) 