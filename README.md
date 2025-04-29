Artificial intelligence (AI) models that can analyse, comprehend, and produce natural language material are known as small language models (SLMs). SLMs are smaller than large language models (LLMs), as their name suggests.

Compared to LLMs, which have hundreds of billions or even trillions of parameters, SLMs have a few million to a few billion. A model learns parameters—internal variables like weights and biases—during training. A machine learning model's behaviour and performance are influenced by these factors.

Compared to their large model counterparts, small language models are more efficient and compact. Because of this, SLMs utilise less memory and processing power, which makes them perfect for environments with limited resources, including edge devices and mobile apps, or even situations where AI inferencing—the process by which a model responds to a user's query—must be carried out offline without a data network.

SLMs are built on top of LLMs. The transformer model is a neural network-based architecture used by small language models, just like large language models. In natural language processing (NLP), transformers have become essential components that serve as the foundation for models such as the generative pre-trained transformer (GPT).

An outline of the transformer architecture is provided below:

● Token positions and semantics are captured by encoders, which convert input sequences into numerical representations known as embeddings.

● Transformers may "focus their attention" on the most significant tokens in the input sequence, independent of their position, thanks to a self-attention mechanism.

● Decoders create the most statistically likely output sequence by utilising this self-attention process in conjunction with the encoders' embeddings.


Primary Libraries and Frameworks
# Create a virtual environment (optional)
python3 -m venv slm_env
source slm_env/bin/activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate accelerate
pip install sentencepiece scikit-learn
pip install peft bitsandbytes # for efficient SLM fine-tuning

# Hugging Face CLI
pip install huggingface_hub
huggingface-cli login


# Model Loading and Fine Tuning

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

model_name = "google/flan-t5-base"  # Replace with your selected SLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Dataset loading (custom medical QA)
from datasets import load_dataset
dataset = load_dataset("med_qa")  # Example or local dataset

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()


# Test Sample:
{
  "question_id": "Q001",
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "answer_gt": "C",
  "model_output": {
    "reasoning_steps": "...",
    "final_answer": "B"
  }, 
  "is_correct": false
}

# Acknowledgement
 



