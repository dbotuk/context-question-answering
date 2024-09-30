import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any


def load_squad_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads the SQuAD dataset from a JSON file and extracts relevant information for training.

    Args:
        filepath (str): Path to the SQuAD dataset file (in JSON format).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary contains a question,
        its context, the answer text, and the start index of the answer in the context.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)['data']
    
    dataset = []

    # Iterate over the articles
    for article in data[:100]:
        # Iterate through paragraphs in each article
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            # Iterate through question-answer pairs
            for qa in paragraph['qas']:
                # Only consider questions that have an answer (i.e., are not impossible)
                if not qa['is_impossible'] and len(qa['answers']) > 0:
                    answer = qa['answers'][0]
                    dataset.append({
                        'id': qa['id'],
                        'question': qa['question'],
                        'context': context,
                        'answer_text': answer['text'],
                        'answer_start': answer['answer_start']
                    })
    return dataset


class SquadDataset(Dataset):
    """
    Custom Dataset class for loading SQuAD data, tokenizing it, and generating training samples for a QA model.

    Args:
        data (List[Dict[str, Any]]): The processed SQuAD dataset.
        tokenizer: Tokenizer used to convert text data into token IDs.
        max_len (int): Maximum length of tokenized inputs (for padding and truncation).
    """
    def __init__(self, data, tokenizer, max_len):
        self.data = data                # List of QA samples with context, question, and answer details
        self.tokenizer = tokenizer      # Tokenizer to process the text data
        self.max_len = max_len          # Max length of tokenized input (context + question)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset and processes it.

        Args:
            idx (int): The index of the sample in the dataset.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing input tensors:
                - 'input_ids': Tensor of tokenized input (context + question).
                - 'attention_mask': Mask for the input (1 for real tokens, 0 for padding tokens).
                - 'start_positions': Tensor indicating the start index of the answer in the context.
                - 'end_positions': Tensor indicating the end index of the answer in the context.
                - 'id': The unique ID of the QA sample.
        """
        item = self.data[idx]
        question = item['question']
        context = item['context']
        start_pos = item['answer_start']
        end_pos = start_pos + len(item['answer_text'])

        # Tokenize the input (context + question)
        inputs = self.tokenizer.encode_plus(
            question, 
            context,
            max_length=self.max_len,        # Maximum length for padding/truncation
            padding='max_length',           # Add padding if the tokenized input is shorter than max_len
            truncation=True,                # Truncate input if it exceeds max_len
            return_tensors="pt"             # Return tensors instead of lists
        )

        # Create start and end position labels for the answer
        start_positions = torch.tensor([start_pos])
        end_positions = torch.tensor([end_pos])

        # Return the input tensors along with start/end positions
        return {
            'input_ids': inputs['input_ids'].squeeze(),                 # Token IDs for input (squeezed to remove extra dimension)
            'attention_mask': inputs['attention_mask'].squeeze(),       # Attention mask (squeezed)
            'start_positions': start_positions,                         # Start index of the answer
            'end_positions': end_positions,                             # End index of the answer
            'id': item['id']                                            # Unique ID of the sample
        }
   