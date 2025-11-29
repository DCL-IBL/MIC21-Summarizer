from transformers import AutoModel, TrainingArguments, Trainer
from datasets import load_dataset

model = AutoModel.from_pretrained("jkralev/mic21_model", trust_remote_code=True, dtype="auto", device_map="cuda")
model.init_components()

data = load_dataset("jkralev/mic21_chess")
data = data.remove_columns(['label','description','size','annotations'])

def data_collator_1(features):
    return {'images':[f['image'] for f in features], 'titles':[f['title'] for f in features]}

training_args = TrainingArguments(
    output_dir="train_chess",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["train"],
    data_collator=data_collator_1
)

trainer.train()