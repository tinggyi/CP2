import os
import re
import json
import requests
import fitz  # PyMuPDF library
from google.cloud import vision

# Lazy-loaded SBERT model to avoid importing torch at module import time
_sbert_model = None
_sbert_model_dir = os.getenv("SBERT_MODEL_PATH", os.path.join(os.getcwd(), "sbert-finetuned"))

# --- 1. SETUP AND AUTHENTICATION ---
# Load Google Cloud credentials from environment variable to avoid committing secrets
# Set an environment variable named `GOOGLE_APPLICATION_CREDENTIALS` that points
# to a local JSON key file (do NOT commit that file into the repo).
credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credential_path:
    if os.path.exists(credential_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    else:
        print(f"⚠️ GOOGLE_APPLICATION_CREDENTIALS is set to '{credential_path}', but file was not found.")
        print("Please set the environment variable to a valid path on your machine.")
else:
    print("⚠️ GOOGLE_APPLICATION_CREDENTIALS not set. Set it to the path of your service-account JSON file.")

# Lazily initialize the Vision client to avoid raising DefaultCredentialsError at import time
vision_client = None

def get_vision_client():
    """Return a cached ImageAnnotatorClient or None if credentials are missing.
    This avoids raising DefaultCredentialsError during import and provides clear
    guidance to the user when credentials are not configured.
    """
    global vision_client
    if vision_client is not None:
        return vision_client
    try:
        vision_client = vision.ImageAnnotatorClient()
        return vision_client
    except Exception as e:
        try:
            from google.auth.exceptions import DefaultCredentialsError
        except Exception:
            DefaultCredentialsError = None

        if DefaultCredentialsError and isinstance(e, DefaultCredentialsError):
            print("⚠️ Google Cloud credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS to the path of your JSON key, or run 'gcloud auth application-default login'.")
            vision_client = None
            return None
        raise

# Use a relative or environment-driven path for the sentence-transformer model
model_dir = os.getenv("SBERT_MODEL_PATH", os.path.join(os.getcwd(), "sbert-finetuned"))
model = None

def get_model():
    global model
    if model is not None:
        return model
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("⚠️ Could not import 'sentence_transformers'. Install it with 'pip install sentence-transformers' and ensure torch is available.")
        raise
    model = SentenceTransformer(model_dir)
    return model

# Load Mistral API key from environment or .env (do NOT hardcode secrets here)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional; environment variables will still be read
    pass

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Warning: MISTRAL_API_KEY not set. Set it in the environment or in a .env file.")

MISTRAL_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- 2. CORE FUNCTIONS ---
def ocr_pdf_to_text(pdf_path):
    """
    Performs OCR on each PDF page and merges consecutive pages intelligently
    so questions spanning pages are kept together.
    """
    print(f" Processing PDF: {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        return ""

    # Ensure Vision client is available before making API calls
    client = get_vision_client()
    if client is None:
        raise RuntimeError("Google Cloud Vision client could not be created: credentials are missing. Set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'.")

    doc = fitz.open(pdf_path)
    full_text_from_pdf = ""
    text_detected = False
    prev_page_text = ""

    for page_num, page in enumerate(doc):
        print(f"  - Reading page {page_num + 1}...")
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        image = vision.Image(content=img_bytes)
        response = client.document_text_detection(image=image)

        if not response.full_text_annotation:
            continue

        page_text = response.full_text_annotation.text.strip()

        # Skip any front matter until we see Question 1
        if not text_detected:
            if "Question" in page_text:
                text_detected = True
            else:
                continue

        # Merge consecutive pages smoothly
        # If the previous page doesn't end with a full stop or "marks)", assume it's cut
        if prev_page_text and not re.search(r'[\.\)\?]$', prev_page_text.strip()):
            full_text_from_pdf = full_text_from_pdf.rstrip() + " " + page_text + "\n\n"
        else:
            full_text_from_pdf += page_text + "\n\n"

        prev_page_text = page_text

    doc.close()
    print("✅ OCR processing complete.")

    # --- Clean up common page footer artifacts ---
    footer_patterns = [
        r'Page\s*\d+\s*of\s*\d+',  # e.g., "Page 2 of 3"
        r'Faculty of Engineering and Technology',
        r'CSC\d{4}/[A-Za-z]+\s*\d{4}\s*Final Examination',
        r'-{3,}',  # horizontal lines like "-----"
    ]

    for pattern in footer_patterns:
        full_text_from_pdf = re.sub(pattern, '', full_text_from_pdf, flags=re.IGNORECASE)

    # Clean up extra blank lines after removal
    full_text_from_pdf = re.sub(r'\n{2,}', '\n\n', full_text_from_pdf)

    return full_text_from_pdf


def extract_questions_and_answers(text):
    import re
    qa_data = {}

    print("\n[DEBUG] Raw text length:", len(text))

    # Only split when "Question X" appears at start of a new line
    main_blocks = re.split(
        r'(?:(?:^|\n|\r)\s*)(?=Question\s*\d+\b)',
        text,
        flags=re.IGNORECASE
    )
    print(f"[DEBUG] Total split blocks: {len(main_blocks)}")

    cleaned_blocks = []
    seen_questions = set()

    # --- Clean and collect unique question blocks ---
    for block in main_blocks:
        block = block.strip()
        if not block:
            continue
        match = re.match(r'Question\s*(\d+)', block, flags=re.IGNORECASE)
        if not match:
            continue
        qnum = match.group(1)
        if qnum in seen_questions:
            print(f"⚠️ Skipping duplicate Question {qnum}")
            continue
        seen_questions.add(qnum)
        cleaned_blocks.append((f"Question {qnum}", block))

    print(f"[DEBUG] Unique questions detected: {seen_questions}")

    # --- Process each question block ---
    for header, main_q_text in cleaned_blocks:
        main_q_num = re.search(r'\d+', header).group()
        print(f"\n[DEBUG] 🧩 Processing {header}")

        # --- FIX: Robustly find the start of the first sub-question (a) or i) ---
        # Look for the start of a letter-based sub-question (a), b), etc.) or roman numerals (i, ii)
        start_sub_q_pattern = r'(?m)^\s*([a-z]\)|\s*i{1,3}[\.\)])'

        sub_start_match = re.search(start_sub_q_pattern, main_q_text, flags=re.IGNORECASE)

        if sub_start_match:
            # Cut off the preamble (Question header, Total marks, Instructions)
            # right before the first sub-question starts.
            main_q_text = main_q_text[sub_start_match.start():]
        else:
            # If no sub-questions are found, we skip this block entirely to avoid corrupting qa_data.
            print(f"⚠️ No sub-questions (a, b, i) found in {header}. Skipping block.")
            continue
        # --- END FIX ---

        # --- Split into subquestions (a), b), etc.) ---
        sub_blocks = re.split(
            r'(?:(?<=^)|(?<=\n)|(?<=\r))\s*([a-z]\))(?=\s*\S)',
            main_q_text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        rebuilt_blocks = []
        for j in range(1, len(sub_blocks), 2):
            label = sub_blocks[j].strip()
            content = sub_blocks[j + 1].strip() if j + 1 < len(sub_blocks) else ""
            rebuilt_blocks.append(label + " " + content)

        # --- Process each sub-question ---
        for sub_block in rebuilt_blocks:
            sub_block = sub_block.strip()
            if not sub_block:
                continue

            parent_match = re.match(r'^([a-z])\)', sub_block, flags=re.IGNORECASE)
            if not parent_match:
                continue
            parent_id = parent_match.group(1).lower()

            # FIX: Update splitter to catch both i) and i.
            nested_splitter = r'(?=\s*\bi{1,3}[\.\)])'
            nested_parts = re.split(nested_splitter, sub_block, flags=re.IGNORECASE)

            # --- CASE 1: Simple part (like Q1b or Q2a/Q2b) ---
            if len(nested_parts) <= 1:
                # Capture marks reliably, setting default to 1
                marks = 1
                marks_match = re.search(r'\(\s*(\d+)\s*mark[s]?\s*\)', sub_block, re.IGNORECASE)
                if marks_match:
                    marks = int(marks_match.group(1))

                parts = re.split(r'\(\s*\d+\s*mark[s]?\s*\)', sub_block, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) < 2:
                    print(f"⚠️ No '(x marks)' found in {main_q_num}{parent_id}")
                    continue

                # 1. Clean Answer: Aggressively strip leading/trailing whitespace.
                answer_text = parts[1].strip()

                # 2. Clean Question: Replace newlines with spaces.
                question_text = parts[0].replace('\n', ' ')

                # 3. Aggressive Strip (Non-Hardcoded Fix for 'occur'):
                answer_text = re.sub(r'^\s*(\w+)\.\s*', '', answer_text, 1, flags=re.IGNORECASE).strip()

                # 4. Final Question Clean (removes prefix 'b)', 'a)', etc.)
                question_text = re.sub(r'^[a-z]\)\s*', '', question_text, 1, flags=re.IGNORECASE).strip()
                # -----------------------------------

                qa_data[f"{main_q_num}{parent_id}"] = {
                    "question": question_text,
                    "answer": answer_text,
                    "marks": marks
                }
                print(f"✅ Captured: Question {main_q_num}{parent_id} ({marks} marks)")

            # --- CASE 2: Nested part (like Q1c_i, Q1c_ii, etc.) ---
            else:
                print(f"ℹ️ Found nested parts in Question {main_q_num}{parent_id}")

                for i, part in enumerate(nested_parts):
                    part = part.strip()
                    if not part:
                        continue

                    # Capture marks reliably, setting default to 1
                    marks = 1
                    marks_match = re.search(r'\(\s*(\d+)\s*mark[s]?\s*\)', part, re.IGNORECASE)
                    if marks_match:
                        marks = int(marks_match.group(1))

                    # CRITICAL FIX APPLIED HERE: Use 'part' instead of 'sub_block'
                    parts = re.split(r'\(\s*\d+\s*mark[s]?\s*\)', part, maxsplit=1, flags=re.IGNORECASE)

                    if len(parts) < 2:
                        # Skip if it's the preamble text that precedes the first 'i)'
                        if i == 0 and not re.match(r'i{1,3}[\.\)]', part, flags=re.IGNORECASE):
                            print(f"ℹ️ Skipping non-question header: {part[:40]}")
                            continue

                        print(f"⚠️ No answer found after marks split in {main_q_num}{parent_id}")
                        continue

                    # --- FIX FOR 'occur' ISSUE (General Cleanup) ---
                    # 1. Clean Answer: Aggressively strip leading/trailing whitespace.
                    answer_text = parts[1].strip()

                    # 2. Clean Question: Replace newlines with spaces.
                    question_text = parts[0].replace('\n', ' ')

                    # 3. Aggressive Strip (Non-Hardcoded Fix for 'occur'):
                    answer_text = re.sub(r'^\s*(\w+)\.\s*', '', answer_text, 1, flags=re.IGNORECASE).strip()
                    # ------------------------------------------------

                    # FIX: Update regex to match both i) and i.
                    roman_match = re.match(r'^(i{1,3})[\.\)]', question_text, flags=re.IGNORECASE)
                    letter_match = re.match(r'^([a-z])\)', question_text, flags=re.IGNORECASE)

                    if roman_match:
                        # Extract the roman numeral part
                        label = roman_match.group(1)
                        key = f"{main_q_num}{parent_id}_{label}"
                    elif letter_match:
                        key = f"{main_q_num}{parent_id}_{letter_match.group(1)}"
                    else:
                        key = f"{main_q_num}{parent_id}_{i}"

                    # Clean up the question text to remove the i) or ii) prefix
                    question_text = re.sub(r'^(i{1,3})[\.\)]\s*', '', question_text, 1, flags=re.IGNORECASE).strip()
                    question_text = re.sub(r'^[a-z]\)\s*', '', question_text, 1, flags=re.IGNORECASE).strip()


                    qa_data[key] = {
                        "question": question_text,
                        "answer": answer_text,
                        "marks": marks
                    }
                    print(f"✅ Captured: Question {key} ({marks} marks)")

    return qa_data

def compute_similarity_and_grade(ref_answer, stu_answer, max_marks=1):
    """Compute cosine similarity and grade (rounded to nearest 0.5)."""
    if not ref_answer or not stu_answer:
        return 0.0, 0.0

    # --- START OF FIX: Ensure max_marks is a valid float ---
    try:
        max_marks = float(max_marks)
        if max_marks <= 0:
            max_marks = 1.0
    except (ValueError, TypeError):
        max_marks = 1.0
    # --- END OF FIX ---

    global _sbert_model, _sbert_model_dir
    try:
        # Import lazily to avoid torch import side-effects at module import time
        from sentence_transformers import SentenceTransformer, util as st_util

        if _sbert_model is None:
            _sbert_model = SentenceTransformer(_sbert_model_dir)
    except Exception as e:
        # If model fails to import/load, log and return safe defaults
        print(f"⚠️ Could not load SBERT model from '{_sbert_model_dir}': {e}")
        return 0.0, 0.0

    try:
        embedding_ref = _sbert_model.encode(ref_answer, convert_to_tensor=True)
        embedding_student = _sbert_model.encode(stu_answer, convert_to_tensor=True)
        similarity_score = st_util.cos_sim(embedding_ref, embedding_student).item()
    except Exception as e:
        print(f"⚠️ Error computing embeddings/similarity: {e}")
        return 0.0, 0.0

    raw_grade = similarity_score * max_marks
    rounded_grade = round(raw_grade * 2) / 2.0
    rounded_grade = max(0, min(rounded_grade, max_marks))
    return similarity_score, rounded_grade


def evaluate_logical_correctness_mistral(question_text, ref_answer, stu_answer, max_marks):
    """
    Evaluate the student's answer using a Chain-of-Thought approach to ensure
    the model first understands the required depth based on max_marks and the reference.
    """

    # --- Final Prompt Modification for Logic Score ---

    prompt = f"""
    You are grading a student's answer for a university exam question worth {max_marks} marks.
    If the answer is correct without writing it in a proper sentence, the answer is still should be perfect match also.

    Question: {question_text}
    Reference Answer: {ref_answer}
    Student Answer: {stu_answer}

    **STRICTNESS RULE:** The student's answer must perfectly align in **sequence, terminology, and content depth**
    with the Reference Answer to achieve 1.0. Any lack of necessary detail or
    substitution of terms (e.g., 'Planning' vs. 'Requirements Analysis') results in a score reduction.

    Rate the student's answer based ONLY on the following:

    0.0 — The answer is factually incorrect, completely irrelevant, or misses an entire required phase/component.
    0.5 — The answer is factually correct based on the {question_text} and {ref_answer}, or applies to answers that
          are correct in concept but insufficient for {max_marks} marks.
    1.0 — The answer is a **PERFECT MATCH** with the {ref_answer}, and the answer is sufficient for {max_marks} marks.

    Respond with ONLY the numeric score (0.0, 0.5, or 1.0), no explanations.
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/devstral-2512:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10
    }

    # --- 2. API Call and Response Handling (No changes here) ---
    try:
        response = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            reply_text = data["choices"][0]["message"]["content"].strip()
        elif "completion" in data:
            reply_text = data["completion"].strip()
        else:
            print(f"⚠️ Warning: Empty or unexpected Mistral response: {data}")
            return 0.0

        # FIX: Robust Numeric Extraction
        match = re.search(r"\b(0\.0|0\.5|1\.0)\b", reply_text.strip())

        if match:
            return float(match.group(1))
        else:
            print(f"⚠️ Warning: Mistral failed to return clean score. Raw text: {reply_text.strip()[:60]}")
            return 0.0

    except requests.exceptions.RequestException as e:
        print(f"❌ Mistral API error during evaluation: {e}")
        return 0.0
    except Exception as e:
        print(f"❌ Unexpected error in logical evaluation: {e}")
        return 0.0


def get_mistral_feedback(question_text, student_answer, correct_answer):
    """
    Asks Mistral (via OpenRouter) to evaluate the student's answer
    and returns only feedback + reasoning.
    """

    system_prompt = """
    You are an expert university examiner.
    Evaluate student answers based ONLY on correctness, completeness,
    clarity, and relevance.
    Always return valid JSON.
    Do NOT include any numeric score.
    """

    user_prompt = f"""
    Evaluate the following exam answer.

    QUESTION:
    {question_text}

    STUDENT ANSWER:
    {student_answer}

    CORRECT ANSWER:
    {correct_answer}

    Return ONLY valid JSON in the following format:
    {{
        "feedback": "<short constructive feedback>",
        "reasoning": "<detailed explanation of why this feedback was given>"
    }}
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/devstral-2512:free",  # Use same model as other function
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }

    try:
        response = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Handle both response structures
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
        elif "completion" in data:
            content = data["completion"]
        else:
            return {
                "feedback": "Error: Unexpected API response format.",
                "reasoning": str(data)
            }

        # Try direct JSON parsing first
        try:
            return json.loads(content)

        except json.JSONDecodeError:
            # Fallback: extract JSON substring using regex
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {
                        "feedback": "Error parsing feedback response.",
                        "reasoning": f"Could not parse JSON from: {match.group(0)[:200]}"
                    }
            else:
                return {
                    "feedback": "Error: Model returned non-JSON response.",
                    "reasoning": f"Raw response: {content[:200]}"
                }

    except requests.exceptions.Timeout:
        return {
            "feedback": "Error: Request timed out.",
            "reasoning": "The API took too long to respond. Please try again."
        }
    except requests.exceptions.HTTPError as e:
        return {
            "feedback": "Error: API request failed.",
            "reasoning": f"HTTP Error: {e.response.status_code}"
        }
    except requests.exceptions.RequestException as e:
        return {
            "feedback": "Error: Network error.",
            "reasoning": str(e)
        }
    except Exception as e:
        print(f"❌ Error calling Mistral: {str(e)}")
        return {
            "feedback": "Error during analysis.",
            "reasoning": str(e)
        }

# --- 3. MAIN WORKFLOW ---
def grade_exam_and_print_report(answer_key_pdf_path, student_answer_pdf_path):
    """
    Main grading function that processes PDFs and generates a grading report.
    Only call this when you want to perform grading, not at module import time.
    """

    answer_key_text = ocr_pdf_to_text(answer_key_pdf_path)
    student_answer_text = ocr_pdf_to_text(student_answer_pdf_path)

    print("\n" + "="*30 + " RAW OCR TEXT FROM STUDENT ANSWER PDF " + "="*30)
    print(student_answer_text)
    print("="*94 + "\n")

    # Parse the text to separate questions and answers
    structured_qa_data = extract_questions_and_answers(answer_key_text)
    student_qa_data = extract_questions_and_answers(student_answer_text)

    print("\n" + "="*25 + " AUTOMATED GRADING REPORT " + "="*25)

    total_marks = 0
    total_scored = 0

    all_keys = sorted(set(structured_qa_data.keys()) | set(student_qa_data.keys()))

    print("\n" + "="*30 + " AUTOMATED GRADING REPORT " + "="*30)

    for key in all_keys:
        ref_data = structured_qa_data.get(key)
        stu_data = student_qa_data.get(key)

        question_text = ref_data.get("question", "[No question text found]") if ref_data else "[No question text found]"
        ref_answer = ref_data.get("answer", "[No reference answer found]") if ref_data else "[No reference answer found]"
        stu_answer = stu_data.get("answer", "[No student answer found]") if stu_data else "[No student answer found]"
        max_marks = ref_data.get("marks", 1) if ref_data else 1

        # Compute similarity score
        similarity, grade = compute_similarity_and_grade(ref_answer, stu_answer, max_marks)

        # Evaluate logical correctness
        logical_score = evaluate_logical_correctness_mistral(
            question_text,
            ref_answer,
            stu_answer,
            max_marks
        )

        # Apply logic-adjusted grade
        if logical_score == 0:
            final_grade = 0
        elif logical_score == 0.5:
            final_grade = (grade + (grade * logical_score)) / 2 # Partially correct
        else:
            final_grade = max_marks

        # --- FIX APPLIED HERE ---
        # The TypeError occurs because max_marks can be None.
        if max_marks is not None:
            # We convert to float/int to handle potential mixing of types,
            # and ensure only numbers are added.
            try:
                total_marks += float(max_marks)
            except ValueError:
                # Handle case where max_marks is a non-numeric string, though less common here.
                print(f"⚠️ Warning: Max marks for question {key} is invalid ('{max_marks}'). Skipping from total marks.")
        else:
            # max_marks was None (due to parsing failure).
            print(f"❌ Error: Max marks for question {key} is missing (None). Skipping from total marks.")
        # ------------------------

        total_scored += final_grade

        # Get AI feedback
        feedback_result = get_mistral_feedback(
            question_text,
            stu_answer,
            ref_answer
        )

        feedback_text = feedback_result.get("feedback", "No feedback")
        reasoning_text = feedback_result.get("reasoning", "No reasoning")

        print(f"\n{'='*80}")
        print(f"Question {key} ({max_marks} marks)")

        print("\nQuestion Text:")
        print(question_text)

        print("\nReference Answer:")
        print(ref_answer)

        print("\nStudent Answer:")
        print(stu_answer)

        print(f"\nCosine Similarity (0–1): {similarity:.4f}")
        print(f"Logical Correctness: {logical_score}")
        print(f"Final Grade: {final_grade:.2f} / {max_marks}")

        print("\n--- AI FEEDBACK (from Mistral) ---")
        print("Feedback:", feedback_text)
        print("\nReasoning:", reasoning_text)

    print(f"\n{'='*80}")
    print(f"TOTAL SCORE: {total_scored:.2f} / {total_marks}")
    print(f"PERCENTAGE: {(total_scored/total_marks)*100:.2f}%")
    print("="*80)


if __name__ == "__main__":
    answer_key_pdf_path = r"D:\CP2_Project\exam_demo2.pdf"
    student_answer_pdf_path = r"D:\CP2_Project\testestla.pdf"

    grade_exam_and_print_report(answer_key_pdf_path, student_answer_pdf_path)
