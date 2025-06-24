Artificial intelligence (AI) models that can analyse, comprehend, and produce natural language material are known as small language models (SLMs). SLMs are smaller than large language models (LLMs), as their name suggests.

Compared to LLMs, which have hundreds of billions or even trillions of parameters, SLMs have a few million to a few billion. A model learns parameters—internal variables like weights and biases—during training. A machine learning model's behaviour and performance are influenced by these factors.

Compared to their large model counterparts, small language models are more efficient and compact. Because of this, SLMs utilise less memory and processing power, which makes them perfect for environments with limited resources, including edge devices and mobile apps, or even situations where AI inferencing—the process by which a model responds to a user's query—must be carried out offline without a data network.

SLMs are built on top of LLMs. The transformer model is a neural network-based architecture used by small language models, just like large language models. In natural language processing (NLP), transformers have become essential components that serve as the foundation for models such as the generative pre-trained transformer (GPT).

An outline of the transformer architecture is provided below:

● Token positions and semantics are captured by encoders, which convert input sequences into numerical representations known as embeddings.

● Transformers may "focus their attention" on the most significant tokens in the input sequence, independent of their position, thanks to a self-attention mechanism.

● Decoders create the most statistically likely output sequence by utilising this self-attention process in conjunction with the encoders' embeddings.


# Model Checklist

Model Loading: HuatuoGPT-o1-8B is typically a causal language model (e.g., based on LLaMA or ChatGLM). So:
Use AutoModelForCausalLM instead of AutoModel
Tokenization doesn't use token type IDs
Input Formatting for MCQA:
Format: "context\n\nQuestion: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:"
All options embedded in the prompt
No pooled output: Use the last token logits to predict the correct answer


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

# HuatuoGPT-o1-8B:
1. Baseline Models

Model	Description
HuatuoGPT-o1-8B	Original large model with 8B parameters; high accuracy but high latency and cost.
GPT-2 Medium	Mid-sized general-purpose model with minimal medical tuning.
BioMed-RoBERTa	Pretrained on biomedical corpora; lacks instruction tuning.
T5-Base (Medical QA)	Fine-tuned on medical QA data; modest performance.
SLM (Ours)	Distilled and instruction-tuned model trained on step-wise prompts.

# eval_results (2).jsonl (Code File):
dataset: The dataset category (e.g., "college_math").
question: The question posed to the model.
response: The full generated explanation by the model.
pred: The model's final prediction/output.
gt_ans: The ground truth answer.
correct: A Boolean indicating if the prediction was correct.

Sample Results:
Question: Simplify: 
− 10 − 4 (n − 5 ) − 10 − 4 (n−5)
Prediction: 10-4n
Ground Truth: $10-4 n$
Correct? True
Question: Solve 

x*y = 90
x* y= 90, (x−5) * (y+1) = 120

(x−5)(y+1)=120
Prediction: 10
Ground Truth: $(45,2),(-10,-9)$
Correct? False
Question: Evaluate (8i * 6 − 7i) * (6 − 7i * 8i)

Prediction: -\frac{56}{85}+\frac{48}{85}i
Ground Truth: \frac{48 i-56}{85}
Correct? True

# Summary of the Statistics
Total Samples Evaluated: 100
Correct Predictions: 48
Incorrect Predictions: 52
Overall Accuracy: 48.00%

# Overall
The visualizing the model's prediction accuracy. As shown:

48% of the predictions were correct.
52% were incorrect.

# Code File:

import json
import sys
import re

import requests

SAVE_TO_FILE = True
OUTPUT_FILE = "huatuo_medqa.jsonl"

def convert_to_huatuo_format(item):
    context = item.get('exp', '').strip()
    question = item['question'].strip()
    options = [item['opa'], item['opb'], item['opc'], item['opd']]
    correct_index = int(item['cop']) - 1
    option_letters = ['A', 'B', 'C', 'D']
    correct_answer = options[correct_index].strip()

    prompt = ""
    if context:
        prompt += f"{context}\n\n"
    prompt += f"Question: {question}\n"
    for i, opt in enumerate(options):
        prompt += f"{option_letters[i]}. {opt.strip()}\n"
    prompt += "\nPlease select the correct option (A, B, C, or D):"

    return {
        "instruction": prompt,
        "input": "",
        "output": f"{option_letters[correct_index]}. {correct_answer}"
    }

def to_elasticsearch_bulk_format(json_objects):
    payload_lines = []
    for obj in json_objects:
        payload_lines.append(json.dumps({"index": {}}))
        payload_lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(payload_lines)

def main():
    huatuo_data = []

    # Read from stdin or file
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            formatted = convert_to_huatuo_format(item)
            huatuo_data.append(formatted)
        except Exception as e:
            print(f"Error parsing line: {e}", file=sys.stderr)

    # Save to file
    if SAVE_TO_FILE:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
            for obj in huatuo_data:
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        print(f"Saved {len(huatuo_data)} entries to {OUTPUT_FILE}")

    # Optional: send to Elasticsearch
    if ELASTIC_SEARCH_URL:
        bulk_payload = to_elasticsearch_bulk_format(huatuo_data)
        response = requests.post(ELASTIC_SEARCH_URL, data=bulk_payload.encode('utf-8'),
                                 headers={"Content-Type": "application/x-ndjson"})
        print(f"Elasticsearch response: {response.status_code} {response.text[:200]}")

if __name__ == "__main__":
    main()

# Results of MedQA

Result MedQA

Metric	Value
Total Questions	50
Correct Answer Matches	50
Accuracy (%)	100.0%

Average Reasoning Length (chars)	140.46 characters

Question (Excerpt)	Correct Answer	Reasoning
21-year-old male with joint pain and dysuria	Ceftriaxone	The patient's symptoms suggest a urogenital infection. The correct antibiotic inhibits bacterial cell wall synthesis, consistent with Ceftriaxone.
5-year-old girl with episodic vomiting	Cyclic vomiting syndrome	The correct answer is Cyclic vomiting syndrome due to episodic vomiting with well periods in between.
40-year-old woman with insomnia and low mood	Trazodone	Trazodone is an antidepressant commonly used to treat depression-related symptoms, including insomnia.
37-year-old diabetic female with flank pain and fever	Obtain a urine analysis and urine culture	The patient's symptoms suggest a urogenital infection. The correct approach is to confirm the diagnosis with a urine analysis.
19-year-old with fruity breath and confusion	Hypoperfusion	The correct answer is Hypoperfusion based on signs of diabetic ketoacidosis and low blood pressure.

# Result MedMCQA

**Result MedMCQA**

**MedQA	Prediction	GroundTruth	Time_Taken**
"Patient has chest pain..."	"Myocardial infarction"	"Myocardial infarction"	1.2s
"Fever and neck stiffness..."	"Meningitis"	"Meningitis"	1.1s


**id	MedQA	Prediction	GroundTruth	Confidence	Time_Taken**
1	Patient reports chest pain.	Myocardial infarction	Myocardial infarction	0.95	1.2
2	High fever and rash.	Dengue	Dengue	0.89	1.0
3	Fatigue and weight loss.	Anemia	Hypothyroidism	0.60	1.3


**ID	MedMCQA	Prediction	GroundTruth	Correct	Confidence	Time_Taken (s)**
1	Patient reports chest pain.	Myocardial infarction	Myocardial infarction	✅	0.95	1.2
2	High fever and rash.	Dengue	Dengue	✅	0.89	1.0
3	Fatigue and weight loss.	Anemia	Hypothyroidism	❌	0.60	1.3

**Metric	Value**
Total Samples	10
Correct Predictions	8
Incorrect Predictions	2
Accuracy (%)	80.0%
Avg. Confidence	0.83
Avg. Time per Record(s)	1.15



# Research Problems / Limitations are find

# Research Problem
In the present outcome, we evaluate the metrices regarding the performance that we achieve while running HuatuoGPT-o1-8B when applied to a dataset comprising college-level mathematical problems. The model is tasked with answering questions that require logical reasoning, symbolic understanding, and accurate problem-solving. The model can generate comprehensive explanations, but the accuracy of the prediction is approximately 48.0% correct and 52.0% wrong, while the accuracy of the prediction is suboptimal. This result shows a significant gap in the ability of the model to consistently generate accurate solutions. A possible factor is the Boolean evaluation mechanism, which answers strictly correct or wrong, without taking into account some arguments or nearly correct output. This raises questions about the reliability of model interpretation skills and the suitability of current assessment metrics in a collection of subtle performance in complex mathematical tasks.

# Hypothesis
•	Boolean precision metric underrealized the true performance of the Huatuogpt-01-8B by not taking into account partially correct mathematical thinking.
•	Fine-tuning of Huatuogpt-01-8b using domain-specific mathematical data can significantly improve the prediction accuracy of mathematical questions at the university level.
•	The 48% accuracy rate of HuatuoGPT-01-8B is inherently affected by the effects of the structure and complexity of mathematical problems as a model limitation.
•	Huatuogpt-01-8b achieves better with arithmetic and algebraic problems than with problems that require logical reasoning or symbolic manipulation in several steps.

# Research Questions
The major research question is mentioned as follows:
1.	What are the limitations of HuatuoGPT-o1-8B in interpreting and solving symbolic and logical discussion tasks in college mathematical dataset?
2.	How HuatuGPT-o1-8B explained by the internal thinking of the model and the probabilistic understanding of mathematical data records?
3.	How can prediction inconsistencies in HuatuoGPT-01-8B be explained by the model’s internal reasoning and probabilistic understanding of mathematical datasets?
4.	What modifications or fine-tuning strategies could improve the performance of HuatuoGPT-01-8B on structured math questions in limited-data environments?



# Proposed Solution

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
 



