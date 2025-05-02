from torch.utils.data import Dataset
import pandas as pd

class HuatuoMCQADataset(Dataset):
    def __init__(self, csv_path: str, use_context: bool = True, return_dict: bool = True):
        self.dataset = pd.read_csv(csv_path)
        self.use_context = use_context
        self.return_dict = return_dict  # Return dictionary format for JSONL-style samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset.iloc[idx]
        context = row['exp'] if self.use_context else ''
        question = row['question']
        options = [row['opa'], row['opb'], row['opc'], row['opd']]
        label_idx = int(row['cop']) - 1
        correct_answer = options[label_idx]
        option_letters = ['A', 'B', 'C', 'D']

        # Build Huatuo-style prompt
        prompt = ""
        if self.use_context and context:
            prompt += f"{context.strip()}\n\n"
        prompt += f"Question: {question.strip()}\n"
        for i, opt in enumerate(options):
            prompt += f"{option_letters[i]}. {opt.strip()}\n"
        prompt += "\nPlease select the correct option (A, B, C, or D):"

        if self.return_dict:
            return {
                "instruction": prompt,
                "input": "",
                "output": f"{option_letters[label_idx]}. {correct_answer.strip()}"
            }
        else:
            return prompt, f"{option_letters[label_idx]}. {correct_answer.strip()}"
