from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import numpy as np

# ----------------------------
# 1️⃣ Load pretrained SBERT
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 2️⃣ Load datasets
# ----------------------------
stsb = load_dataset("sentence-transformers/stsb")
sickr = load_dataset("mteb/sickr-sts")
sts12 = load_dataset("mteb/sts12-sts")

train_data_stsb = stsb['train']
train_data_sickr = sickr['test']  # using test split as train
train_data_sts12 = sts12['train']

val_data = stsb['validation']

# ----------------------------
# 3️⃣ Helper function: convert to InputExample
# ----------------------------
def create_input_examples(dataset):
    examples = []
    for row in dataset:
        examples.append(InputExample(
            texts=[row["sentence1"], row["sentence2"]],
            label=float(row["score"]) / 5.0  # normalize to [0,1]
        ))
    return examples

# ----------------------------
# 4️⃣ Prepare weighted training samples
# ----------------------------
train_samples = []
# Weight STS-B higher by repeating 3x
train_samples.extend(create_input_examples(train_data_stsb) * 3)
train_samples.extend(create_input_examples(train_data_sickr))
train_samples.extend(create_input_examples(train_data_sts12))

val_samples = create_input_examples(val_data)

# ----------------------------
# 5️⃣ DataLoader & loss
# ----------------------------
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)

# ----------------------------
# 6️⃣ Training with validation monitoring
# ----------------------------
epochs = 12
warmup_steps = 200

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': 1e-5},
        show_progress_bar=True,
        use_amp=True  # mixed-precision, optional
    )

    # ----------------------------
    # Validation after each epoch
    # ----------------------------
    embeddings1 = model.encode([x.texts[0] for x in val_samples], convert_to_tensor=True)
    embeddings2 = model.encode([x.texts[1] for x in val_samples], convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings1, embeddings2).diagonal().cpu().numpy()
    true_scores = np.array([x.label for x in val_samples])

    mse = mean_squared_error(true_scores, cos_scores)
    pearson_corr, _ = pearsonr(true_scores, cos_scores)
    spearman_corr, _ = spearmanr(true_scores, cos_scores)

    print(f"Validation MSE: {mse:.4f} | Pearson: {pearson_corr:.4f} | Spearman: {spearman_corr:.4f}")

# ----------------------------
# 7️⃣ Save fine-tuned model
# ----------------------------
model.save("output/sbert-stsb-sickr-sts12-model")