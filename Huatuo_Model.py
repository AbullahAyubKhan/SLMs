from transformers import AutoTokenizer, AutoModelForCausalLM
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import functools
import math
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class HuatuoMCQAModel(pl.LightningModule):
    def __init__(self, model_name_or_path, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.args = args
        self.batch_size = args['batch_size']
        self.ce_loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def prepare_dataset(self, train_dataset, val_dataset, test_dataset=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset if test_dataset is not None else val_dataset

    def format_prompt(self, context, question, options):
        choices = ['A', 'B', 'C', 'D']
        formatted_options = "\n".join([f"{c}. {opt}" for c, opt in zip(choices, options)])
        prompt = f"{context}\n\nQuestion: {question}\n{formatted_options}\nAnswer:"
        return prompt

    def process_batch(self, batch, tokenizer, max_len=512):
        inputs = []
        labels = []
        for data in batch:
            if len(data) == 4:
                context, question, options, answer = data
            else:
                question, options, answer = data
                context = ""

            prompt = self.format_prompt(context, question, options)
            input_ids = tokenizer(prompt, truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')['input_ids'][0]
            label_id = ord('A') + answer  # Assuming answer is index 0-3 → A-D
            inputs.append(input_ids)
            labels.append(label_id)

        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)
        return {"input_ids": inputs}, labels

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        last_token_logits = logits[:, -1, :]  # get logits of last token
        return last_token_logits  # shape: [batch_size, vocab_size]

    def compute_loss(self, logits, labels):
        answer_tokens = [self.tokenizer.convert_tokens_to_ids(c) for c in ['A', 'B', 'C', 'D']]
        logits = logits[:, answer_tokens]  # [batch_size, 4]
        return self.ce_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        input_ids = inputs['input_ids'].to(self.device)
        labels = labels.to(self.device)
        logits = self.forward(input_ids)
        loss = self.compute_loss(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        input_ids = inputs['input_ids'].to(self.device)
        labels = labels.to(self.device)
        logits = self.forward(input_ids)
        loss = self.compute_loss(logits, labels)
        preds = torch.argmax(logits[:, [self.tokenizer.convert_tokens_to_ids(c) for c in ['A', 'B', 'C', 'D']]], dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args['learning_rate'], eps=1e-8)
        total_steps = (len(self.train_dataset) // self.args['batch_size']) * self.args['num_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler,
                          collate_fn=functools.partial(self.process_batch, tokenizer=self.tokenizer, max_len=self.args['max_len']))

    def val_dataloader(self):
        sampler = SequentialSampler(self.val_dataset)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler,
                          collate_fn=functools.partial(self.process_batch, tokenizer=self.tokenizer, max_len=self.args['max_len']))

    def test_dataloader(self):
        sampler = SequentialSampler(self.test_dataset)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=sampler,
                          collate_fn=functools.partial(self.process_batch, tokenizer=self.tokenizer, max_len=self.args['max_len']))
