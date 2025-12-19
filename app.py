import os
import sys

# Disable Streamlit file watcher to avoid conflicts with PyTorch
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

# Suppress PyTorch warnings
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Automated Short Answer Grading System", layout="wide")

# Enhanced Custom CSS
st.markdown("""
<style>
/* Main app styling */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
}

/* Modal overlay */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.modal-content {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    min-width: 300px;
}

.spinner {
    border: 5px solid rgba(255, 255, 255, 0.3);
    border-top: 5px solid white;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-text {
    font-size: 18px;
    font-weight: 600;
    color: white;
    margin-top: 15px;
}

.loading-subtext {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.9);
    margin-top: 10px;
}

/* Score badge styling */
.score-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 16px;
    margin: 10px 0;
}

.score-excellent {
    background: #4CAF50;
    color: white;
}

.score-good {
    background: #4CAF50;
    color: white;
}

.score-average {
    background: #F4D03F;
    color: white;
}

.score-poor {
    background: #E74C3C;
    color: white;
}

.score-fail {
    background: #E74C3C;
    color: white;
}

/* Feedback styling */
.feedback-box {
    background: linear-gradient(135deg, #e0f2f7 0%, #f0e6ff 100%);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin-top: 15px;
}

.feedback-label {
    font-weight: bold;
    color: #667eea;
    margin-bottom: 8px;
}

.feedback-content {
    color: #333;
    line-height: 1.6;
}

/* Header styling */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Text area styling for better readability */
textarea {
    font-family: inherit !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 10px !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}

/* Info and warning boxes styling */
.stAlert {
    line-height: 1.6 !important;
    word-wrap: break-word !important;
}

/* Student ID dialog modal */
.student-dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
}

.student-dialog-content {
    background: white;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    min-width: 400px;
    max-width: 500px;
}

.dialog-title {
    font-size: 24px;
    font-weight: bold;
    color: #667eea;
    margin-bottom: 20px;
    text-align: center;
}

.dialog-input {
    width: 100%;
    padding: 12px;
    font-size: 16px;
    border: 2px solid #667eea;
    border-radius: 8px;
    margin-bottom: 20px;
}

.dialog-buttons {
    display: flex;
    gap: 10px;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "both_uploaded" not in st.session_state:
    st.session_state.both_uploaded = False
if "extracted" not in st.session_state:
    st.session_state.extracted = False
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "extracting" not in st.session_state:
    st.session_state.extracting = False
if "modules_loaded" not in st.session_state:
    st.session_state.modules_loaded = False
if "feedback_cache" not in st.session_state:
    st.session_state.feedback_cache = {}
if "show_id_dialog" not in st.session_state:
    st.session_state.show_id_dialog = False
if "current_view" not in st.session_state:
    st.session_state.current_view = "welcome"  # welcome, extracted, results, records
if "edited_grades" not in st.session_state:
    st.session_state.edited_grades = {}
if "grade_edit_mode" not in st.session_state:
    st.session_state.grade_edit_mode = {}
if "current_record_id" not in st.session_state:
    st.session_state.current_record_id = None

# Database functions
def init_database():
    """Initialize SQLite database for storing grading results"""
    conn = sqlite3.connect('grading_results.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS student_grades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            grading_date TEXT NOT NULL,
            total_score REAL,
            max_marks REAL,
            percentage REAL,
            avg_similarity REAL,
            avg_logic REAL,
            question_details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def clear_database():
    """Clear all records from the database"""
    conn = sqlite3.connect('grading_results.db')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS student_grades')
    conn.commit()
    conn.close()
    # Reinitialize the database
    init_database()

def save_to_database(student_id, results, stats):
    """Save grading results to database and return the record ID"""
    conn = sqlite3.connect('grading_results.db')
    c = conn.cursor()

    # Convert results to JSON string for storage
    import json
    question_details = json.dumps(results)

    c.execute('''
        INSERT INTO student_grades
        (student_id, grading_date, total_score, max_marks, percentage, avg_similarity, avg_logic, question_details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        student_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        float(stats['total_scored']),
        float(stats['total_marks']),
        float(stats['percentage']),
        float(stats['avg_similarity']),
        float(stats['avg_logic']),
        question_details
    ))

    record_id = c.lastrowid  # Get the ID of the inserted record
    conn.commit()
    conn.close()
    return record_id

def update_database_record(record_id, results, stats):
    """Update an existing database record with new grades"""
    conn = sqlite3.connect('grading_results.db')
    c = conn.cursor()

    # Convert results to JSON string for storage
    import json
    question_details = json.dumps(results)

    c.execute('''
        UPDATE student_grades
        SET total_score = ?, max_marks = ?, percentage = ?,
            avg_similarity = ?, avg_logic = ?, question_details = ?
        WHERE id = ?
    ''', (
        float(stats['total_scored']),
        float(stats['total_marks']),
        float(stats['percentage']),
        float(stats['avg_similarity']),
        float(stats['avg_logic']),
        question_details,
        record_id
    ))

    conn.commit()
    conn.close()

def get_all_records():
    """Retrieve all grading records from database"""
    conn = sqlite3.connect('grading_results.db')
    df = pd.read_sql_query("SELECT * FROM student_grades ORDER BY grading_date DESC", conn)
    conn.close()
    return df

# Initialize database
init_database()

# Uncomment the line below to clear the database (run once then comment it back)
# clear_database()

# Lazy load modules only when needed
def load_modules():
    """Load heavy modules only when first needed"""
    if not st.session_state.modules_loaded:
        import tempfile
        from Capstone2_Project import (
            ocr_pdf_to_text,
            extract_questions_and_answers,
            compute_similarity_and_grade,
            evaluate_logical_correctness_mistral,
            get_mistral_feedback
        )
        st.session_state.modules_loaded = True
        st.session_state.ocr_pdf_to_text = ocr_pdf_to_text
        st.session_state.extract_questions_and_answers = extract_questions_and_answers
        st.session_state.compute_similarity_and_grade = compute_similarity_and_grade
        st.session_state.evaluate_logical_correctness_mistral = evaluate_logical_correctness_mistral
        st.session_state.get_mistral_feedback = get_mistral_feedback
        st.session_state.tempfile = tempfile

# Helper function to get score badge HTML
def get_score_badge(score, max_marks):
    try:
        max_marks = float(max_marks)
    except (ValueError, TypeError):
        max_marks = 1.0

    if max_marks == 0:
        percentage = 0.0
    else:
        percentage = (score / max_marks) * 100

    if percentage >= 90:
        badge_class = "score-excellent"
        label = "Excellent"
    elif percentage >= 75:
        badge_class = "score-good"
        label = "Good"
    elif percentage >= 50:
        badge_class = "score-average"
        label = "Average"
    elif percentage >= 10:
        badge_class = "score-poor"
        label = "Poor"
    else:
        badge_class = "score-fail"
        label = "Needs Improvement"

    display_max_marks = max_marks if max_marks > 0 else 1

    return f'<div class="score-badge {badge_class}">🎯 Score: {score:.2f}/{display_max_marks:.0f} ({percentage:.1f}%) - {label}</div>'

# Main title
st.title("📘 Automated Short Answer Grading System")
st.markdown("---")

# Create two-column layout
left_col, right_col = st.columns([1, 2])

# --- LEFT COLUMN: File Upload and Controls ---
with left_col:
    st.header("📤 Upload & Process")

    ref_pdf = st.file_uploader(
        "Reference Answer Paper",
        type=["pdf"],
        help="Upload the exam paper with reference answers",
        key="ref_uploader"
    )

    stu_pdf = st.file_uploader(
        "Student Answer Paper",
        type=["pdf"],
        help="Upload the student's answer sheet",
        key="stu_uploader"
    )

    # Clear previous results when files change
    if ref_pdf and stu_pdf:
        current_files = (ref_pdf.name, stu_pdf.name)
        if "previous_files" not in st.session_state:
            st.session_state.previous_files = current_files
        elif st.session_state.previous_files != current_files:
            # Files have changed - clear all cached results
            st.session_state.extracted = False
            st.session_state.confirmed = False
            if "ref_qa" in st.session_state:
                del st.session_state.ref_qa
            if "stu_qa" in st.session_state:
                del st.session_state.stu_qa
            if "grading_results" in st.session_state:
                del st.session_state.grading_results
            if "feedback_cache" in st.session_state:
                st.session_state.feedback_cache = {}
            if "edited_answers" in st.session_state:
                del st.session_state.edited_answers
            if "edit_mode" in st.session_state:
                del st.session_state.edit_mode
            if "edited_grades" in st.session_state:
                st.session_state.edited_grades = {}
            st.session_state.previous_files = current_files

    # Check upload status
    if ref_pdf and stu_pdf:
        st.success("✅ Both files uploaded!")
        st.session_state.both_uploaded = True

        # Extract button
        if st.button("🔍 Extract & Process", use_container_width=True, type="primary", disabled=st.session_state.get("extracting", False)):
            st.session_state.extracted = False
            st.session_state.confirmed = False
            st.session_state.extracting = True
            st.session_state.loading_phase = "reading"
            st.session_state.current_view = "welcome"  # Reset view during extraction
            # Store file contents before rerun
            st.session_state.ref_pdf_bytes = ref_pdf.read()
            st.session_state.stu_pdf_bytes = stu_pdf.read()
            st.rerun()

        # Show extraction in progress with modal
        if st.session_state.get("extracting"):
            # Display modal overlay based on loading phase
            modal_placeholder = st.empty()

            loading_phase = st.session_state.get("loading_phase", "reading")

            if loading_phase == "reading":
                with modal_placeholder.container():
                    st.markdown("""
                    <div class="modal-overlay">
                        <div class="modal-content">
                            <div class="spinner"></div>
                            <div class="loading-text">⚙️ Initializing AI Models...</div>
                            <div class="loading-subtext">Loading machine learning components (this may take a moment on first run)</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Load modules
                load_modules()
                st.session_state.loading_phase = "processing"
                st.rerun()

            elif loading_phase == "processing":
                with modal_placeholder.container():
                    st.markdown("""
                    <div class="modal-overlay">
                        <div class="modal-content">
                            <div class="spinner"></div>
                            <div class="loading-text">🔄 Processing Documents...</div>
                            <div class="loading-subtext">Extracting and analyzing content from PDFs</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Save uploaded files to temporary files using stored bytes
                with st.session_state.tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_ref:
                    tmp_ref.write(st.session_state.ref_pdf_bytes)
                    ref_path = tmp_ref.name

                with st.session_state.tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_stu:
                    tmp_stu.write(st.session_state.stu_pdf_bytes)
                    stu_path = tmp_stu.name

                try:
                    ref_text = st.session_state.ocr_pdf_to_text(ref_path)
                    stu_text = st.session_state.ocr_pdf_to_text(stu_path)

                    st.session_state.ref_qa = st.session_state.extract_questions_and_answers(ref_text)
                    st.session_state.stu_qa = st.session_state.extract_questions_and_answers(stu_text)
                    st.session_state.extracted = True
                    st.session_state.current_view = "extracted"  # Auto switch to extracted view

                finally:
                    # Clean up temporary files and stored bytes
                    os.unlink(ref_path)
                    os.unlink(stu_path)
                    st.session_state.extracting = False
                    # Clear stored file bytes to free memory
                    if 'ref_pdf_bytes' in st.session_state:
                        del st.session_state.ref_pdf_bytes
                    if 'stu_pdf_bytes' in st.session_state:
                        del st.session_state.stu_pdf_bytes
                    if 'loading_phase' in st.session_state:
                        del st.session_state.loading_phase
                    modal_placeholder.empty()
                    st.rerun()

        # Grade button (only show after extraction and not yet graded)
        if st.session_state.get("extracted") and not st.session_state.get("confirmed"):
            if st.button("✅ Grade Answers", use_container_width=True, type="primary"):
                st.session_state.show_id_dialog = True
                st.rerun()

        # View Extracted Q&A button (only show after extraction)
        if st.session_state.get("extracted"):
            if st.button("📋 View Extracted Q&A", use_container_width=True):
                st.session_state.current_view = "extracted"
                st.rerun()

        # View Graded Results button (only show after grading is complete)
        if st.session_state.get("confirmed") and "grading_results" in st.session_state:
            if st.button("🎯 View Graded Results", use_container_width=True):
                st.session_state.current_view = "results"
                st.rerun()

        # Save Edited Grades button (only show after grading)
        if st.session_state.get("confirmed") and st.session_state.get("current_record_id"):
            if st.button("💾 Save Final Grades", use_container_width=True, type="primary"):
                # Recalculate statistics with edited grades
                if "grading_results" in st.session_state:
                    results = st.session_state.grading_results

                    # Update grades with edited values
                    for result in results:
                        q_num = result['Question']
                        if q_num in st.session_state.edited_grades:
                            result['Grade'] = st.session_state.edited_grades[q_num]

                    df = pd.DataFrame(results)
                    stats = {
                        'total_scored': df["Grade"].sum(),
                        'total_marks': df["Max Marks"].sum(),
                        'percentage': (df["Grade"].sum() / df["Max Marks"].sum()) * 100,
                        'avg_similarity': df["Similarity"].mean(),
                        'avg_logic': df["Logic Score"].mean()
                    }

                    # Update the specific database record
                    update_database_record(st.session_state.current_record_id, results, stats)
                    st.success("✅ Grades updated successfully!")
                    st.rerun()
    else:
        st.info("👆 Please upload both PDF files to begin")
        st.session_state.both_uploaded = False
        st.session_state.current_view = "welcome"

    # View All Records button (always available)
    if st.button("📊 View All Records", use_container_width=True):
        st.session_state.current_view = "records"
        st.rerun()

# Student ID Dialog Modal
@st.dialog("Enter Student ID")
def student_id_dialog():
    st.write("Please enter the student's ID to save grading results:")
    student_id = st.text_input("Student ID:", key="dialog_student_id", placeholder="e.g., S12345678")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm & Grade", use_container_width=True, type="primary"):
            if student_id.strip():
                st.session_state.current_student_id = student_id.strip()
                st.session_state.confirmed = True
                st.session_state.show_id_dialog = False
                st.session_state.current_view = "results"  # Auto navigate to results
                st.rerun()
            else:
                st.error("Please enter a student ID!")
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_id_dialog = False
            st.rerun()

# Show dialog if needed - moved before showing modal to prevent flicker
if st.session_state.get("show_id_dialog"):
    student_id_dialog()
    # If dialog was just closed via confirmation, don't show anything below
    if not st.session_state.get("show_id_dialog"):
        st.stop()

# --- RIGHT COLUMN: Display Area ---
with right_col:
    # Route display based on current_view
    current_view = st.session_state.get("current_view", "welcome")

    if current_view == "welcome":
        st.header("📋 Welcome")
        st.info("Upload both PDF files and click 'Extract & Process' to begin.")

    elif current_view == "records":
        st.header("📊 All Grading Records")
        records = get_all_records()
        if not records.empty:
            # Display summary table (without question details)
            display_df = records[['student_id', 'grading_date', 'total_score', 'max_marks', 'percentage', 'avg_similarity', 'avg_logic']].copy()

            # Ensure numeric columns are properly formatted
            display_df['total_score'] = pd.to_numeric(display_df['total_score'], errors='coerce').round(2)
            display_df['max_marks'] = pd.to_numeric(display_df['max_marks'], errors='coerce').round(0)
            display_df['percentage'] = pd.to_numeric(display_df['percentage'], errors='coerce').round(2)
            display_df['avg_similarity'] = pd.to_numeric(display_df['avg_similarity'], errors='coerce').round(3)
            display_df['avg_logic'] = pd.to_numeric(display_df['avg_logic'], errors='coerce').round(2)

            display_df.columns = ['Student ID', 'Date', 'Score', 'Max Marks', 'Percentage (%)', 'Avg Similarity', 'Avg Logic']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Download all records
            csv = records.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download All Records (CSV)",
                csv,
                "all_grading_records.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No grading records found.")

    elif current_view == "extracted":
        st.header("📋 Extracted Questions & Answers")
        st.markdown("Review the extracted content before grading. Click 'Edit' to correct any OCR errors.")

        # Initialize edited answers in session state if not exists
        if "edited_answers" not in st.session_state:
            st.session_state.edited_answers = {}
            for key in st.session_state.stu_qa.keys():
                st.session_state.edited_answers[key] = st.session_state.stu_qa[key].get("answer", "")

        # Initialize edit mode tracking
        if "edit_mode" not in st.session_state:
            st.session_state.edit_mode = {}

        for key in sorted(st.session_state.ref_qa.keys()):
            ref = st.session_state.ref_qa[key]
            stu = st.session_state.stu_qa.get(key, {"answer": "[No student answer found]"})

            with st.expander(f"❓ Question {key} ({ref.get('marks', 1)} marks)", expanded=False):
                # Edit button in top right
                edit_col1, edit_col2 = st.columns([5, 1])
                with edit_col2:
                    edit_key = f"edit_mode_q{key}"
                    if edit_key not in st.session_state.edit_mode:
                        st.session_state.edit_mode[edit_key] = False

                    if st.session_state.edit_mode[edit_key]:
                        if st.button("💾", key=f"save_btn_q{key}", use_container_width=True, type="primary"):
                            st.session_state.edit_mode[edit_key] = False
                            st.rerun()
                    else:
                        if st.button("✏️", key=f"edit_btn_q{key}", use_container_width=True):
                            st.session_state.edit_mode[edit_key] = True
                            st.rerun()

                with edit_col1:
                    st.markdown("**📝 Question:**")
                    st.write(ref['question'])

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✓ Reference Answer:**")
                    st.info(ref['answer'])
                with col_b:
                    st.markdown("**📄 Student Answer:**")
                    current_answer = st.session_state.edited_answers.get(key, stu['answer'])

                    # Show editable text area only if in edit mode
                    if st.session_state.edit_mode.get(f"edit_mode_q{key}", False):
                        edited_answer = st.text_area(
                            "Edit student answer:",
                            value=current_answer,
                            height=200,
                            key=f"textarea_q{key}",
                            label_visibility="collapsed"
                        )
                        st.session_state.edited_answers[key] = edited_answer
                    else:
                        # Display as read-only
                        st.warning(current_answer)

    elif current_view == "results":
        st.header("🎯 Grading Results")

        # Ensure modules are loaded
        if not st.session_state.modules_loaded:
            load_modules()

        # Check if results are already cached - ONLY compute once
        if "grading_results" not in st.session_state:
            results = []
            total = len(st.session_state.ref_qa)
            progress = st.progress(0)
            status_text = st.empty()

            for i, key in enumerate(sorted(st.session_state.ref_qa.keys())):
                status_text.text(f"Grading Question {key}... ({i+1}/{total})")

                ref = st.session_state.ref_qa[key]
                # Use edited answer if available, otherwise use original
                stu_answer = st.session_state.edited_answers.get(key, st.session_state.stu_qa.get(key, {"answer": ""})["answer"])

                sim, grade = st.session_state.compute_similarity_and_grade(ref["answer"], stu_answer, ref["marks"])
                logic = st.session_state.evaluate_logical_correctness_mistral(ref["question"], ref["answer"], stu_answer, ref["marks"])

                # Apply the SAME logic as Capstone2_Project.py
                if logic == 0:
                    final = 0
                elif logic == 0.5:
                    final = (grade + (grade * logic)) / 2
                else:  # logic == 1.0
                    final = ref["marks"]

                final_grade_rounded = round(final * 2) / 2.0
                final_grade_capped = min(ref["marks"], final_grade_rounded)

                results.append({
                    "Question": key,
                    "Similarity": round(sim, 3),
                    "Logic Score": logic,
                    "Grade": round(final_grade_capped, 2),
                    "Max Marks": ref["marks"],
                    "ref_question": ref["question"],
                    "ref_answer": ref["answer"],
                    "stu_answer": stu_answer
                })

                progress.progress((i + 1) / total)

            status_text.empty()
            progress.empty()

            # Cache results in session state
            st.session_state.grading_results = results

            # Initialize edited grades with computed grades
            for result in results:
                if result['Question'] not in st.session_state.edited_grades:
                    st.session_state.edited_grades[result['Question']] = result['Grade']

            # Calculate statistics and save to database
            df = pd.DataFrame(results)
            stats = {
                'total_scored': df["Grade"].sum(),
                'total_marks': df["Max Marks"].sum(),
                'percentage': (df["Grade"].sum() / df["Max Marks"].sum()) * 100,
                'avg_similarity': df["Similarity"].mean(),
                'avg_logic': df["Logic Score"].mean()
            }

            # Save to database and store the record ID
            record_id = save_to_database(st.session_state.current_student_id, results, stats)
            st.session_state.current_record_id = record_id
            # Auto switch to results view
            st.session_state.current_view = "results"
        else:
            results = st.session_state.grading_results

        # Calculate statistics using edited grades
        total_scored = sum(st.session_state.edited_grades.get(r['Question'], r['Grade']) for r in results)
        total_marks = sum(r['Max Marks'] for r in results)
        percentage = (total_scored / total_marks) * 100
        avg_similarity = sum(r['Similarity'] for r in results) / len(results)
        avg_logic = sum(r['Logic Score'] for r in results) / len(results)

        st.success("🎉 Grading Complete!")

        # Show student ID
        if "current_student_id" in st.session_state:
            st.info(f"**Student ID:** {st.session_state.current_student_id}")

        st.markdown("---")

        # Display each question with editable grade
        st.subheader("📊 Detailed Question-by-Question Results")
        for result in results:
            q_num = result['Question']
            current_grade = st.session_state.edited_grades.get(q_num, result['Grade'])

            with st.expander(f"❓ Question {q_num} - {result['Max Marks']} marks", expanded=True):
                # Display score badge with current (possibly edited) grade
                current_grade = st.session_state.edited_grades.get(q_num, result['Grade'])
                st.markdown(get_score_badge(current_grade, result['Max Marks']), unsafe_allow_html=True)

                # Question text
                st.markdown("---")
                st.markdown("**📝 Question:**")
                st.write(result['ref_question'])

                # Answers in columns
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✓ Reference Answer:**")
                    st.info(result['ref_answer'])
                with col_b:
                    st.markdown("**📄 Student Answer:**")
                    st.warning(result['stu_answer'])

                # Metrics in columns with editable grade
                st.markdown("---")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns([1.5, 1.5, 1.5, 0.5])

                with metric_col1:
                    st.metric("Similarity Score", f"{result['Similarity']:.3f}")

                with metric_col2:
                    st.metric("Logic Score", f"{result['Logic Score']:.1f}")

                with metric_col3:
                    # Initialize edit mode for this question
                    grade_edit_key = f"grade_edit_q{q_num}"
                    if grade_edit_key not in st.session_state.grade_edit_mode:
                        st.session_state.grade_edit_mode[grade_edit_key] = False

                    if st.session_state.grade_edit_mode[grade_edit_key]:
                        # Show text input field for manual typing
                        new_grade_str = st.text_input(
                            "Final Grade",
                            value=str(current_grade),
                            key=f"grade_input_q{q_num}",
                            placeholder=f"0.0 - {result['Max Marks']}"
                        )
                        # Validate and save the grade immediately
                        try:
                            new_grade = float(new_grade_str)
                            if 0 <= new_grade <= result['Max Marks']:
                                # Save immediately when value changes
                                if new_grade != current_grade:
                                    st.session_state.edited_grades[q_num] = new_grade
                            else:
                                st.error(f"Grade must be between 0 and {result['Max Marks']}")
                        except ValueError:
                            if new_grade_str.strip():  # Only show error if user typed something
                                st.error("Please enter a valid number")
                    else:
                        # Show read-only display as metric
                        display_val = f"{current_grade:.2f}/{result['Max Marks']}"
                        if current_grade != result['Grade']:
                            st.metric("Final Grade 🔄", display_val)
                        else:
                            st.metric("Final Grade", display_val)

                with metric_col4:
                    st.write("")  # Spacing
                    st.write("")  # Spacing
                    if st.session_state.grade_edit_mode[grade_edit_key]:
                        if st.button("💾", key=f"save_grade_q{q_num}", use_container_width=True):
                            # Ensure the grade is saved before closing edit mode
                            input_key = f"grade_input_q{q_num}"
                            if input_key in st.session_state:
                                try:
                                    final_grade = float(st.session_state[input_key])
                                    if 0 <= final_grade <= result['Max Marks']:
                                        st.session_state.edited_grades[q_num] = final_grade
                                except (ValueError, KeyError):
                                    pass
                            st.session_state.grade_edit_mode[grade_edit_key] = False
                            st.rerun()
                    else:
                        if st.button("✏️", key=f"edit_grade_q{q_num}", use_container_width=True):
                            st.session_state.grade_edit_mode[grade_edit_key] = True
                            st.rerun()

                # Feedback button and display
                st.markdown("---")

                feedback_key = f"feedback_q{result['Question']}"

                if st.button(
                    "💬 Receive Feedback",
                    key=f"btn_{result['Question']}",
                    use_container_width=True
                ):
                    with st.spinner("🤖 Generating AI feedback..."):
                        feedback = st.session_state.get_mistral_feedback(
                            result['ref_question'],
                            result['stu_answer'],
                            result['ref_answer']
                        )
                        st.session_state.feedback_cache[feedback_key] = feedback

                # Display cached feedback if available
                if feedback_key in st.session_state.feedback_cache:
                    feedback_data = st.session_state.feedback_cache[feedback_key]

                    st.markdown(f"""
                    <div class="feedback-box">
                        <div class="feedback-label">📝 AI Feedback:</div>
                        <div class="feedback-content">{feedback_data.get('feedback', 'N/A')}</div>
                        <br>
                        <div class="feedback-label">🔍 Reasoning:</div>
                        <div class="feedback-content">{feedback_data.get('reasoning', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Statistics Section with Visual Charts
        st.subheader("📈 Overall Performance Statistics")

        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Score", f"{total_scored:.2f}/{total_marks}", f"{percentage:.1f}%")
        with col2:
            st.metric("Average Similarity", f"{avg_similarity:.3f}")
        with col3:
            st.metric("Average Logic Score", f"{avg_logic:.2f}")

        st.markdown("---")

        # Create two columns for charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Bar chart: Grades per question (with edited grades)
            st.markdown("##### 📊 Grades per Question")
            chart_data = pd.DataFrame({
                'Question': [f"Q{r['Question']}" for r in results],
                'Grade': [st.session_state.edited_grades.get(r['Question'], r['Grade']) for r in results],
                'Max Marks': [r['Max Marks'] for r in results]
            })
            st.bar_chart(chart_data.set_index('Question'))

        with chart_col2:
            # Line chart: Similarity and Logic Scores
            st.markdown("##### 📈 Similarity & Logic Trends")
            trend_data = pd.DataFrame({
                'Question': [f"Q{r['Question']}" for r in results],
                'Similarity': [r['Similarity'] for r in results],
                'Logic Score': [r['Logic Score'] for r in results]
            })
            st.line_chart(trend_data.set_index('Question'))

        st.markdown("---")

        # Download button
        download_df = pd.DataFrame([
            {
                "Question": r['Question'],
                "Similarity": r['Similarity'],
                "Logic Score": r['Logic Score'],
                "AI Grade": r['Grade'],
                "Final Grade": st.session_state.edited_grades.get(r['Question'], r['Grade']),
                "Max Marks": r['Max Marks']
            }
            for r in results
        ])
        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Results (CSV)",
            csv,
            f"{st.session_state.current_student_id}_grading_results.csv",
            "text/csv",
            use_container_width=True
        )