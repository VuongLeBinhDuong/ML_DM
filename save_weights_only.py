import torch
import sys

def save_weights_only(input_path, output_path, model_key='model_state_dict'):
    checkpoint = torch.load(input_path, map_location='cpu')
    # If checkpoint is a dict and has model_state_dict, use it; else try 'state_dict', else use as is
    if isinstance(checkpoint, dict) and model_key in checkpoint:
        state_dict = checkpoint[model_key]
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    torch.save({'model_state_dict': state_dict}, output_path)
    print(f"Saved weights only to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        # Default paths, edit as needed
        input_path = "demo/models/bert_classifier.pth"
        output_path = "demo/models/bert_classifier_weights_only.pth"
    save_weights_only(input_path, output_path) 