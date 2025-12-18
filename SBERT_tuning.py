from sentence_transformers import SentenceTransformer, util

# Load the fine-tuned model
model = SentenceTransformer(r"D:\CP2_Project\sbert-finetuned")

reference_answer = "The process of photosynthesis converts sunlight into chemical energy."
student_answer = "Photosynthesis changes sunlight into energy stored in plants."

# 3️⃣ Encode sentences
embedding_ref = model.encode(reference_answer, convert_to_tensor=True)
embedding_student = model.encode(student_answer, convert_to_tensor=True)

# 4️⃣ Compute cosine similarity
similarity_score = util.cos_sim(embedding_ref, embedding_student).item()  # scalar

# 5️⃣ Optionally scale to 0-5 score
predicted_grade = similarity_score * 5  # since you normalized to [0,1] during training

print(f"Cosine similarity (0-1): {similarity_score:.4f}")
print(f"Predicted grade (0-5): {predicted_grade:.2f}")