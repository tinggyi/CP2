from sentence_transformers import SentenceTransformer, util

# Load the fine-tuned model
model = SentenceTransformer(r"D:\CP2_Project\sbert-finetuned")

reference_answer = "The capital of France is Paris."
student_answer = "The capital of France is the United Kingdom."

# Encode sentences
embedding_ref = model.encode(reference_answer, convert_to_tensor=True)
embedding_student = model.encode(student_answer, convert_to_tensor=True)

# Compute cosine similarity
similarity_score = util.cos_sim(embedding_ref, embedding_student).item()

print(f"Cosine similarity (0–1): {similarity_score:.4f}")