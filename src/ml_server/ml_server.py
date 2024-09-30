import mlflow
import torch

from flask import Flask, jsonify, request
from mlflow import MlflowClient
from os import environ
from transformers import DistilBertTokenizer
from typing import Tuple, Dict, Any


app = Flask(__name__)

# Set up MLflow client and environment variables
client = MlflowClient()
mlflow_tracking_url = environ.get('MLFLOW_TRACKING_URI')
run_id = "30d41a3da0fb46c2b8a4958789eca187"
model_artifact_name = "bert_model"


def load_model_and_params(run_id: str, artifact_name: str) -> Tuple[torch.nn.Module, DistilBertTokenizer, Dict[str, Any]]:
    """
    Load the model, tokenizer, and parameters from the latest run in the specified MLflow experiment.

    Returns:
        Tuple[torch.nn.Module, DistilBertTokenizer, Dict[str, Any]]:
            - Loaded PyTorch model.
            - Tokenizer corresponding to the model.
            - Dictionary of model parameters from the run.
    Raises:
        ValueError: If the experiment or run is not found.
    """
    model_uri = f"runs:/{run_id}/{artifact_name}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    # Fetch the run's parameters and load the tokenizer
    run = mlflow.get_run(run_id)
    params = run.data.params
    tokenizer = DistilBertTokenizer.from_pretrained(params['tokenizer'])

    return model, tokenizer, params


try:
    # Load model, tokenizer, and parameters on application startup
    model, tokenizer, params = load_model_and_params(run_id, model_artifact_name)
    app.logger.info(f"Model and tokenizer loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model or parameters: {e}")
    raise e


@app.route('/answer', methods=['POST'])
def predict_answer() -> Tuple[Dict[str, Any], int]:
    """
    Endpoint to handle POST requests for question-answer prediction.

    Expects JSON input with "context" and "question" fields.

    Returns:
        Tuple[Dict[str, Any], int]:
            - A dictionary with the predicted answer and a message indicating success or failure.
            - HTTP status code.
    """
    try:
        # Parse input data from the request
        request_data = request.get_json()
        context = request_data["context"]
        question = request_data["question"]

        # Select the device (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Encode the inputs using the tokenizer
        inputs = tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,                # Adds special tokens like [CLS] and [SEP]
            max_length=int(params['max_len']),      # Maximum length of tokenized input
            truncation=True,                        # Truncate if the input exceeds max length
            padding="max_length",                   # Pad input to max length
            return_tensors="pt"                     # Return as PyTorch tensors
        )

        # Move input tensors to the appropriate device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Perform inference in no_grad mode (for efficiency)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Extract the start and end logits from model outputs
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get the predicted start and end positions of the answer
        start_pred = torch.argmax(start_logits, dim=-1).item()
        end_pred = torch.argmax(end_logits, dim=-1).item()

        # Compute max logits to estimate the "no answer" probability
        start_logit_max = torch.max(start_logits, dim=-1).values.item()
        end_logit_max = torch.max(end_logits, dim=-1).values.item()
        no_answer_prob = 1 - (start_logit_max + end_logit_max) / 2.0

        # Check if the model predicts "no answer" or an invalid prediction
        if no_answer_prob > float(params['no_answer_threshold']) or start_pred > end_pred:
            return jsonify({'message': 'No answer!'}), 200

        # Decode the predicted tokens into the final answer string
        pred_answer = tokenizer.decode(input_ids[0][start_pred:end_pred + 1], skip_special_tokens=True)

        return jsonify({'result': pred_answer, 'message': 'Successful!'}), 200

    except Exception as e:
        # Log errors and return a 500 response if something goes wrong
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=environ.get('ML_SERVER_PORT'))
