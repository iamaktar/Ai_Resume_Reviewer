import os
import re
import json
import streamlit as st
import pdfplumber
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# ==========================================
# MODULE 1: PDF EXTRACTION & CLEANING
# ==========================================
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extracts and cleans text from an uploaded PDF file using pdfplumber.
    """
    try:
        extracted_pages = []
        with pdfplumber.open(uploaded_file) as pdf:
            if not pdf.pages:
                raise ValueError("The uploaded PDF contains no pages.")
            
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)
        
        full_text = "\n".join(extracted_pages)
        
        # Clean up excessive whitespace and non-standard line breaks
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        
        if not cleaned_text:
            raise ValueError(
                "No selectable text found in the PDF. Please ensure the file is not a scanned image."
            )
            
        return cleaned_text

    except Exception as e:
        raise RuntimeError(f"Failed to read PDF file: {str(e)}")


# ==========================================
# MODULE 2: GEMINI AI ANALYSIS
# ==========================================
def analyze_resume_with_gemini(resume_text: str, jd_text: str, api_key: str) -> dict:
    """
    Sends the extracted resume text and job description to the Gemini API
    and returns a structured JSON evaluation.
    """
    genai.configure(api_key=api_key)

    system_instruction = (
        "You are an expert Technical Recruiter and Applicant Tracking System (ATS) specialist. "
        "Compare the candidate's resume against the target Job Description objectively. "
        "You must respond strictly in valid JSON format matching this exact structure:\n"
        "{\n"
        '  "ats_score": <integer between 0 and 100>,\n'
        '  "missing_keywords": ["keyword1", "keyword2", ...],\n'
        '  "profile_summary": "<2-3 sentence objective evaluation of candidate fit>",\n'
        '  "actionable_feedback": ["bullet 1", "bullet 2", "bullet 3", "bullet 4"]\n'
        "}\n"
        "Rules:\n"
        "1. Be rigorous with the ats_score based on hard skills, tools, and experience alignment.\n"
        "2. Keep missing_keywords strictly to technical skills, tools, certifications, or domain terms present in the JD but absent in the resume.\n"
        "3. Provide 3 to 5 highly specific, actionable improvements in actionable_feedback."
    )

    # Configure model to return strict JSON
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
    )

    user_prompt = f"""
    === TARGET JOB DESCRIPTION ===
    {jd_text}

    === CANDIDATE RESUME ===
    {resume_text}
    """

    try:
        response = model.generate_content(user_prompt)
        
        if not response or not response.text:
            raise ValueError("Received an empty response from the Gemini model.")
            
        result_json = json.loads(response.text)
        
        # Validate expected keys exist with safe fallbacks
        return {
            "ats_score": int(result_json.get("ats_score", 0)),
            "missing_keywords": list(result_json.get("missing_keywords", [])),
            "profile_summary": str(result_json.get("profile_summary", "No summary provided.")),
            "actionable_feedback": list(result_json.get("actionable_feedback", []))
        }

    except google_exceptions.ResourceExhausted:
        raise RuntimeError("API rate limit exceeded (Quota Exhausted). Please wait a moment and try again.")
    except google_exceptions.InvalidArgument as e:
        raise RuntimeError(f"Invalid API request or API Key: {str(e)}")
    except json.JSONDecodeError:
        raise RuntimeError("Failed to parse the AI output into structured data. Please try running the analysis again.")
    except Exception as e:
        raise RuntimeError(f"Gemini API Error: {str(e)}")


# ==========================================
# MODULE 3: UI RENDERING
# ==========================================
def render_dashboard(analysis_data: dict):
    """
    Renders the evaluation metrics, missing keywords, and feedback on the Streamlit UI.
    """
    st.divider()
    st.subheader("Analysis Results")

    score = max(0, min(100, analysis_data["ats_score"]))
    
    # Top row: ATS Score gauge & Profile Summary
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="ATS Match Score", value=f"{score}%")
        st.progress(score / 100.0)
        if score >= 80:
            st.success("Strong Match")
        elif score >= 55:
            st.warning("Moderate Match")
        else:
            st.error("Low Match")

    with col2:
        st.markdown("**Profile Summary**")
        st.info(analysis_data["profile_summary"])

    st.divider()

    # Bottom row: Missing Keywords & Actionable Feedback
    col_kw, col_fb = st.columns([1, 1])

    with col_kw:
        st.markdown("**Missing Keywords**")
        missing_kw = analysis_data["missing_keywords"]
        if missing_kw:
            # Render keywords as clean visual badges using HTML/Markdown
            badges_html = " ".join(
                [
                    f"<span style='display:inline-block; background-color:#ff4b4b22; "
                    f"color:#ff4b4b; border:1px solid #ff4b4b; border-radius:12px; "
                    f"padding:4px 10px; margin:4px 2px; font-size:0.85rem; font-weight:500;'>"
                    f"{kw}</span>"
                    for kw in missing_kw
                ]
            )
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.success("No major keywords missing!")

    with col_fb:
        st.markdown("**Actionable Feedback**")
        feedback_list = analysis_data["actionable_feedback"]
        if feedback_list:
            for item in feedback_list:
                st.markdown(f"- {item}")
        else:
            st.write("No specific feedback items returned.")


def main():
    st.set_page_config(
        page_title="AI Resume Reviewer",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 AI Resume Reviewer & ATS Matcher")
    st.caption("Upload your PDF resume and paste a target Job Description to evaluate your ATS fit.")

    # Load API key silently from .env
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    # Sidebar: Resume Upload Only
    with st.sidebar:
        st.header("1. Upload Resume")
        uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
        
        if uploaded_file:
            st.success(f"Loaded: {uploaded_file.name}")

    # Main Area: Job Description Input
    st.subheader("2. Target Job Description")
    jd_text = st.text_area(
        "Paste the full Job Description (JD) below:",
        height=220,
        placeholder="Paste the role responsibilities, required skills, and qualifications here..."
    )

    analyze_btn = st.button("Analyze Resume", type="primary", use_container_width=True)

    if analyze_btn:
        # Input Validations
        if not api_key:
            st.error("GEMINI_API_KEY not found. Please add it to your .env file and restart the app.")
            return

        if not uploaded_file:
            st.warning("Please upload a PDF resume in the sidebar before analyzing.")
            return

        if not jd_text or len(jd_text.strip()) < 20:
            st.warning("Please paste a valid Job Description (at least 20 characters).")
            return

        # Execution Pipeline
        with st.spinner("Extracting text from PDF and running ATS analysis..."):
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
                
                with st.expander("Preview Extracted Resume Text"):
                    st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))

                analysis_result = analyze_resume_with_gemini(
                    resume_text=resume_text,
                    jd_text=jd_text.strip(),
                    api_key=api_key
                )
                
                render_dashboard(analysis_result)

            except Exception as err:
                st.error(str(err))


if __name__ == "__main__":
    main()