import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

# ---------------------- Page Config ----------------------

st.set_page_config(
    page_title="AI Resume Screening & ATS Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Custom CSS for Better Styling ----------------------

st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-container h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .header-container p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    /* Skills section styling */
    .skills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .skill-tag {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .skill-tag-missing {
        background: #ffebee;
        color: #c62828;
    }
    
    .skill-tag-found {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #333;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Input styling */
    .stTextArea, .stFileUploader {
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError {
        border-radius: 8px;
    }
    
    /* Results container */
    .results-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------- Skill Database ----------------------

TECHNICAL_SKILLS = [
    # Programming Languages
    "python", "java", "javascript", "c++", "c#", "go", "rust", "ruby", "php",
    
    # Data & ML
    "sql", "pandas", "numpy", "machine learning", "deep learning", "nlp",
    "tensorflow", "scikit-learn", "pytorch", "keras",
    
    # Web Frameworks
    "streamlit", "flask", "django", "fastapi", "react", "angular", "vue",
    
    # Web Technologies
    "html", "css", "rest api", "graphql", "webapi",
    
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "cassandra",
    
    # DevOps & Cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "ci/cd",
    
    # Data Visualization
    "power bi", "tableau", "matplotlib", "seaborn", "plotly",
    
    # Data Science
    "data analysis", "data visualization", "data science", "data mining",
    "data cleaning", "data wrangling", "feature engineering",
    "classification", "regression", "exploratory data analysis",
    
    # Tools & Others
    "git", "github", "excel", "powerpoint", "microsoft office",
    "communication", "teamwork", "problem solving", "statistics",
    "spring boot"
]

# ---------------------- Functions ----------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        raise Exception(f"Unable to read PDF: {str(e)}")
    return text

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_ats_score(resume_text: str, job_description: str) -> float:
    """Calculate ATS score using TF-IDF and cosine similarity."""
    documents = [resume_text, job_description]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)
    similarity = cosine_similarity(matrix)[0][1]
    return similarity * 100

def find_skills(resume_text: str, skill_list: List[str]) -> List[str]:
    """Find skills present in resume."""
    clean_resume = resume_text.lower()
    clean_resume = clean_resume.replace("-", " ")
    clean_resume = clean_resume.replace("/", " ")
    
    found = []
    for skill in skill_list:
        skill_lower = skill.lower()
        skill_lower = skill_lower.replace("-", " ")
        
        if re.search(r"\b" + re.escape(skill_lower) + r"\b", clean_resume):
            found.append(skill)
    
    return sorted(list(set(found)))

def find_missing_skills(resume_text: str, job_description: str, skill_list: List[str]) -> List[str]:
    """Find skills mentioned in job description but missing from resume."""
    jd_lower = job_description.lower()
    resume_lower = resume_text.lower()
    
    missing = []
    for skill in skill_list:
        skill_lower = skill.lower()
        if skill_lower in jd_lower and skill_lower not in resume_lower:
            missing.append(skill)
    
    return sorted(list(set(missing)))

def get_ats_color(score: float) -> str:
    """Get color based on ATS score."""
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    else:
        return "🔴"

def get_ats_feedback(score: float) -> str:
    """Get feedback based on ATS score."""
    if score >= 85:
        return "Excellent match! Your resume aligns well with the job description."
    elif score >= 70:
        return "Good match! Consider adding some missing keywords."
    elif score >= 50:
        return "Fair match. Review the job description and add relevant skills."
    else:
        return "Low match. Consider revising your resume to better match the job."

# ---------------------- Header ----------------------

st.markdown("""
    <div class="header-container">
        <h1>📄 AI Resume Screening & ATS Analyzer</h1>
        <p>Optimize your resume for Applicant Tracking Systems and job requirements</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- Sidebar ----------------------

with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    ### How to use:
    1. **Upload** your resume in PDF format
    2. **Paste** the job description
    3. **Click** "Analyze Resume"
    
    ### What you'll get:
    - **ATS Score**: Similarity between your resume and job description
    - **Found Skills**: Skills you have that match the job
    - **Missing Skills**: Skills to highlight or develop
    - **Suggestions**: Personalized recommendations
    
    ### Tips:
    - Use keywords from the job description
    - Include both technical and soft skills
    - Format your resume clearly
    - Quantify your achievements
    """)
    
    st.divider()
    st.markdown("**💡 Pro Tip:** Most recruiters use ATS systems to screen resumes. Aim for 70%+ ATS score!")

# ---------------------- Input Section ----------------------

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="section-header">📑 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF format)",
        type=["pdf"],
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<div class="section-header">📝 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste the job description",
        height=250,
        placeholder="Paste the complete job description here...",
        label_visibility="collapsed"
    )

# ---------------------- Analyze Button ----------------------

analyze_button = st.button("🚀 Analyze Resume", use_container_width=True)

# ---------------------- Analysis Logic ----------------------

if analyze_button:
    # Validation
    if uploaded_file is None:
        st.error("⚠️ Please upload a PDF resume to continue.", icon="❌")
        st.stop()

    if job_description.strip() == "":
        st.error("⚠️ Please paste the job description to continue.", icon="❌")
        st.stop()

    # Extract and process
    with st.spinner("🔄 Analyzing your resume..."):
        try:
            # Extract text
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # Clean text
            clean_resume = clean_text(resume_text)
            clean_jd = clean_text(job_description)
            
            # Calculate metrics
            ats_score = calculate_ats_score(clean_resume, clean_jd)
            found_skills = find_skills(clean_resume, TECHNICAL_SKILLS)
            missing_skills = find_missing_skills(clean_resume, job_description, TECHNICAL_SKILLS)
            
        except Exception as e:
            st.error(f"❌ Error processing resume: {str(e)}")
            st.stop()

    # ---------------------- Results Section ----------------------

    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # ---------------------- ATS Score Card ----------------------
    
    st.markdown('<div class="section-header">📊 ATS Score Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    
    with col1:
        st.metric(
            label="ATS Score",
            value=f"{ats_score:.1f}%",
            delta=None,
            label_visibility="visible"
        )
    
    with col2:
        st.metric(
            label="Skills Found",
            value=len(found_skills),
            label_visibility="visible"
        )
    
    with col3:
        st.metric(
            label="Skills Missing",
            value=len(missing_skills),
            label_visibility="visible"
        )
    
    # ATS Feedback
    feedback = get_ats_feedback(ats_score)
    status_icon = get_ats_color(ats_score)
    st.info(f"{status_icon} {feedback}", icon="ℹ️")
    
    # ---------------------- Skills Found Section ----------------------
    
    st.markdown('<div class="section-header">✅ Skills Found in Your Resume</div>', unsafe_allow_html=True)
    
    if found_skills:
        # Display skills in columns for better layout
        skill_cols = st.columns(4)
        for idx, skill in enumerate(found_skills):
            with skill_cols[idx % 4]:
                st.success(f"✔ {skill.title()}")
        
        st.caption(f"Total: {len(found_skills)} matching skills found")
    else:
        st.warning("No matching skills found. Consider adding more relevant keywords.")
    
    # ---------------------- Missing Skills Section ----------------------
    
    st.markdown('<div class="section-header">⚠️ Skills to Highlight or Develop</div>', unsafe_allow_html=True)
    
    if missing_skills:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display missing skills in a better format
            skill_cols = st.columns(4)
            for idx, skill in enumerate(missing_skills):
                with skill_cols[idx % 4]:
                    st.error(f"✗ {skill.title()}")
            
            st.caption(f"Total: {len(missing_skills)} skills mentioned in job description but not in resume")
        
        with col2:
            st.info(f"📌 **Priority:** Review these {len(missing_skills)} skills and add them if you have experience.")
    else:
        st.success("🎉 Great! All required skills are present in your resume!")
    
    # ---------------------- Suggestions Section ----------------------
    
    st.markdown('<div class="section-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
    
    with st.expander("📌 See Detailed Suggestions", expanded=True):
        if ats_score >= 80:
            st.success("✅ **Excellent!** Your resume is well-optimized. Focus on:")
            st.markdown("""
            - Adding quantifiable metrics and achievements
            - Using action verbs to describe your experience
            - Customizing your summary for each application
            """)
        
        elif ats_score >= 60:
            st.info("💼 **Good Foundation** - Here's how to improve:")
            st.markdown("""
            - Add more technical keywords from the job description
            - Mention specific tools and technologies you've used
            - Reorganize sections to highlight relevant experience first
            """)
        
        else:
            st.warning("⚠️ **Needs Improvement** - Take these actions:")
            st.markdown("""
            - Mirror the language and keywords from the job description
            - Add missing skills if you have experience with them
            - Reorganize your resume to emphasize relevant skills
            """)
        
        if missing_skills:
            st.subheader("Missing Skills - Action Items:")
            for i, skill in enumerate(missing_skills[:10], 1):  # Show top 10
                st.markdown(f"**{i}. {skill.title()}**")
                st.caption("Add to your resume if you have experience with this skill")
    
    # ---------------------- Resume Preview ----------------------
    
    st.divider()
    st.markdown('<div class="section-header">📄 Resume Preview</div>', unsafe_allow_html=True)
    
    with st.expander("Show Resume Text (First 3000 characters)"):
        preview_text = resume_text[:3000]
        st.text(preview_text)
        if len(resume_text) > 3000:
            st.caption(f"... (Total: {len(resume_text)} characters)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------------------- Download Report Option ----------------------
    
    st.divider()
    
    # Create a simple text report
    report = f"""
    RESUME ANALYSIS REPORT
    =====================
    
    ATS SCORE: {ats_score:.1f}%
    
    FEEDBACK: {feedback}
    
    SKILLS FOUND ({len(found_skills)}):
    {', '.join([s.title() for s in found_skills])}
    
    SKILLS TO ADD ({len(missing_skills)}):
    {', '.join([s.title() for s in missing_skills[:15]])}
    
    RECOMMENDATIONS:
    - Review and incorporate missing skills if applicable
    - Optimize keyword placement in your resume
    - Ensure clear formatting for better ATS parsing
    - Include both technical and soft skills
    - Use specific metrics and achievements
    """
    
    st.download_button(
        label="📥 Download Analysis Report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain"
    )

# ---------------------- Footer ----------------------

st.divider()
st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.9rem; margin-top: 3rem;">
        <p>💡 <strong>Remember:</strong> This tool helps optimize your resume for ATS systems. 
        However, the best resumes are personalized, well-written, and showcase your unique value.</p>
        <p style="margin-top: 1rem;">© 2024 AI Resume Analyzer | Built with Streamlit</p>
        <p style="margin-top: 0.5rem; font-size: 0.85rem;">Developed by <strong>Karamthot Anil Naik</strong></p>
    </div>
    """, unsafe_allow_html=True)