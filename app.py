import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from collections import Counter
import PyPDF2
import docx
import io
from datetime import datetime
import base64
import json

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Page configuration
st.set_page_config(
    page_title="SkillMirror - AI Career Feedback",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffc107;
        margin: 1rem 0;
    }
    .skill-match {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
    }
    .skill-gap {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Skill categories and keywords
SKILL_CATEGORIES = {
    'Programming Languages': [
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 
        'go', 'rust', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'typescript', 'c'
    ],
    'Web Development': [
        'react', 'angular', 'vue', 'node.js', 'nodejs', 'express', 'django', 'flask', 'spring',
        'laravel', 'rails', 'asp.net', 'bootstrap', 'jquery', 'webpack', 'babel', 'next.js'
    ],
    'Data Science & Analytics': [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'matplotlib',
        'seaborn', 'plotly', 'tableau', 'power bi', 'jupyter', 'spark', 'hadoop', 'hive'
    ],
    'Cloud & DevOps': [
        'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'git',
        'github', 'gitlab', 'terraform', 'ansible', 'nginx', 'apache', 'linux', 'unix'
    ],
    'Databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch', 'oracle',
        'sqlite', 'dynamodb', 'firebase', 'nosql', 'sql server'
    ],
    'Mobile Development': [
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'swift', 'kotlin',
        'objective-c', 'cordova', 'ionic'
    ],
    'Soft Skills': [
        'leadership', 'communication', 'teamwork', 'problem solving', 'project management',
        'agile', 'scrum', 'collaboration', 'analytical', 'creative', 'adaptable'
    ]
}

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(text, skill_categories):
    """Extract skills from text based on predefined categories"""
    text = preprocess_text(text)
    found_skills = {}
    
    for category, skills in skill_categories.items():
        found_skills[category] = []
        for skill in skills:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text):
                found_skills[category].append(skill)
    
    return found_skills

def calculate_ats_score(resume_text):
    """Calculate ATS compatibility score"""
    score = 0
    feedback = []
    
    # Check for common ATS-friendly elements
    if len(resume_text) > 200:
        score += 10
    else:
        feedback.append("Resume seems too short")
    
    # Check for section headers
    common_sections = ['experience', 'education', 'skills', 'summary', 'objective']
    found_sections = sum(1 for section in common_sections if section in resume_text.lower())
    score += found_sections * 5
    
    if found_sections < 3:
        feedback.append("Missing common resume sections")
    
    # Check for contact information patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    if re.search(email_pattern, resume_text):
        score += 10
    else:
        feedback.append("No email address found")
    
    if re.search(phone_pattern, resume_text):
        score += 10
    else:
        feedback.append("No phone number found")
    
    # Check for dates (experience)
    date_pattern = r'\b(19|20)\d{2}\b'
    if re.search(date_pattern, resume_text):
        score += 10
    else:
        feedback.append("No dates found (work experience)")
    
    # Check for action verbs
    action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'designed', 
                   'built', 'improved', 'increased', 'reduced', 'achieved']
    found_verbs = sum(1 for verb in action_verbs if verb in resume_text.lower())
    score += min(found_verbs * 2, 20)
    
    if found_verbs < 3:
        feedback.append("Consider using more action verbs")
    
    # Check for quantifiable achievements
    number_pattern = r'\b\d+%?\b'
    numbers_found = len(re.findall(number_pattern, resume_text))
    score += min(numbers_found * 2, 15)
    
    if numbers_found < 3:
        feedback.append("Add more quantifiable achievements")
    
    return min(score, 100), feedback

def analyze_job_match(resume_skills, job_description):
    """Analyze how well resume matches job requirements"""
    job_skills = extract_skills(job_description, SKILL_CATEGORIES)
    
    matches = {}
    gaps = {}
    total_job_skills = 0
    total_matches = 0
    
    for category in SKILL_CATEGORIES.keys():
        job_category_skills = set(job_skills.get(category, []))
        resume_category_skills = set(resume_skills.get(category, []))
        
        matches[category] = list(job_category_skills.intersection(resume_category_skills))
        gaps[category] = list(job_category_skills - resume_category_skills)
        
        total_job_skills += len(job_category_skills)
        total_matches += len(matches[category])
    
    match_percentage = (total_matches / total_job_skills * 100) if total_job_skills > 0 else 0
    
    return matches, gaps, match_percentage

def generate_recommendations(ats_score, match_percentage, skill_gaps, ats_feedback):
    """Generate personalized recommendations"""
    recommendations = []
    
    # ATS recommendations
    if ats_score < 70:
        recommendations.append({
            'type': 'ATS Optimization',
            'priority': 'High',
            'recommendation': 'Your resume needs ATS optimization. ' + '; '.join(ats_feedback[:3])
        })
    
    # Skill gap recommendations
    if match_percentage < 60:
        top_gaps = []
        for category, gaps in skill_gaps.items():
            if gaps:
                top_gaps.extend(gaps[:2])  # Top 2 gaps per category
        
        if top_gaps:
            recommendations.append({
                'type': 'Skill Development',
                'priority': 'High',
                'recommendation': f'Focus on learning: {", ".join(top_gaps[:5])}'
            })
    
    # General recommendations
    recommendations.extend([
        {
            'type': 'Content Enhancement',
            'priority': 'Medium',
            'recommendation': 'Add more quantifiable achievements with specific numbers and percentages'
        },
        {
            'type': 'Keywords',
            'priority': 'Medium',
            'recommendation': 'Include more industry-specific keywords from the job description'
        },
        {
            'type': 'Format',
            'priority': 'Low',
            'recommendation': 'Ensure consistent formatting and use standard section headers'
        }
    ])
    
    return recommendations[:5]  # Return top 5 recommendations

# Main App
def main():
    st.markdown('<h1 class="main-header">🎯 SkillMirror</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Resume Analysis & Career Feedback</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload your resume** (PDF or DOCX)
        2. **Paste the job description** you're applying for
        3. **Get instant AI feedback** on:
           - ATS compatibility score
           - Skill matching analysis
           - Personalized recommendations
           - Career insights
        """)
        
        st.header("🎯 Features")
        st.markdown("""
        - **ATS Score Analysis**
        - **Skill Gap Identification**
        - **Job Match Percentage**
        - **Interactive Visualizations**
        - **Actionable Recommendations**
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx'],
            help="Upload your resume in PDF or DOCX format"
        )
        
        resume_text = ""
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_text_from_docx(uploaded_file)
            
            if resume_text:
                st.success(f"✅ Resume uploaded successfully! ({len(resume_text)} characters)")
                with st.expander("Preview Resume Text"):
                    st.text_area("Resume Content", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200, disabled=True)
    
    with col2:
        st.header("💼 Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=300,
            placeholder="Paste the full job description including requirements, responsibilities, and preferred qualifications..."
        )
    
    # Analysis button
    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        if not resume_text:
            st.error("Please upload a resume first!")
        elif not job_description:
            st.error("Please paste a job description!")
        else:
            with st.spinner("Analyzing your resume..."):
                # Extract skills
                resume_skills = extract_skills(resume_text, SKILL_CATEGORIES)
                
                # Calculate ATS score
                ats_score, ats_feedback = calculate_ats_score(resume_text)
                
                # Analyze job match
                skill_matches, skill_gaps, match_percentage = analyze_job_match(resume_skills, job_description)
                
                # Generate recommendations
                recommendations = generate_recommendations(ats_score, match_percentage, skill_gaps, ats_feedback)
            
            st.success("Analysis complete! 🎉")
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="ATS Compatibility Score",
                    value=f"{ats_score}/100",
                    delta=f"{'Good' if ats_score >= 70 else 'Needs Improvement'}"
                )
            
            with col2:
                st.metric(
                    label="Job Match Percentage",
                    value=f"{match_percentage:.1f}%",
                    delta=f"{'Strong Match' if match_percentage >= 60 else 'Moderate Match' if match_percentage >= 40 else 'Weak Match'}"
                )
            
            with col3:
                total_skills = sum(len(skills) for skills in resume_skills.values())
                st.metric(
                    label="Skills Identified",
                    value=str(total_skills),
                    delta="From your resume"
                )
            
            # Visualizations
            st.header("📈 Visual Analysis")
            
            # Skills comparison chart
            col1, col2 = st.columns(2)
            
            with col1:
                # Skill matches by category
                match_data = []
                for category in SKILL_CATEGORIES.keys():
                    matches = len(skill_matches.get(category, []))
                    gaps = len(skill_gaps.get(category, []))
                    if matches > 0 or gaps > 0:
                        match_data.append({
                            'Category': category,
                            'Matches': matches,
                            'Gaps': gaps
                        })
                
                if match_data:
                    df_matches = pd.DataFrame(match_data)
                    fig = px.bar(
                        df_matches.melt(id_vars=['Category'], var_name='Type', value_name='Count'),
                        x='Category',
                        y='Count',
                        color='Type',
                        title='Skill Matches vs Gaps by Category',
                        color_discrete_map={'Matches': '#4CAF50', 'Gaps': '#f44336'}
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ATS score breakdown (simulate components)
                ats_components = {
                    'Contact Information': min(20, ats_score * 0.2),
                    'Section Structure': min(25, ats_score * 0.25),
                    'Keywords': min(30, ats_score * 0.3),
                    'Formatting': min(25, ats_score * 0.25)
                }
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(ats_components.keys()),
                        y=list(ats_components.values()),
                        marker_color='#4CAF50'
                    )
                ])
                fig.update_layout(
                    title='ATS Score Breakdown',
                    yaxis_title='Score',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed skill analysis
            st.header("🎯 Skill Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Matching Skills")
                for category, matches in skill_matches.items():
                    if matches:
                        st.write(f"**{category}:**")
                        for skill in matches:
                            st.markdown(f'<span class="skill-match">{skill}</span>', unsafe_allow_html=True)
                        st.write("")
            
            with col2:
                st.subheader("❌ Missing Skills")
                for category, gaps in skill_gaps.items():
                    if gaps:
                        st.write(f"**{category}:**")
                        for skill in gaps:
                            st.markdown(f'<span class="skill-gap">{skill}</span>', unsafe_allow_html=True)
                        st.write("")
            
            # Recommendations
            st.header("💡 Personalized Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                priority_color = {
                    'High': '#f44336',
                    'Medium': '#ff9800',
                    'Low': '#4CAF50'
                }
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <h4>#{i} {rec['type']} 
                    <span style="color: {priority_color[rec['priority']]}; font-size: 0.8em;">
                    [{rec['priority']} Priority]
                    </span></h4>
                    <p>{rec['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ATS Feedback
            if ats_feedback:
                st.header("⚠️ ATS Optimization Tips")
                for feedback in ats_feedback:
                    st.markdown(f"""
                    <div class="warning-box">
                        <p>⚡ {feedback}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export results
            st.header("📥 Export Results")
            
            # Prepare data for export
            results_data = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ats_score': ats_score,
                'match_percentage': match_percentage,
                'skill_matches': skill_matches,
                'skill_gaps': skill_gaps,
                'recommendations': recommendations,
                'ats_feedback': ats_feedback
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_data = json.dumps(results_data, indent=2)
                st.download_button(
                    label="📄 Download Analysis (JSON)",
                    data=json_data,
                    file_name=f"skillmirror_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Summary report
                summary_report = f"""
SKILLMIRROR ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SUMMARY ===
ATS Compatibility Score: {ats_score}/100
Job Match Percentage: {match_percentage:.1f}%
Total Skills Identified: {sum(len(skills) for skills in resume_skills.values())}

=== TOP RECOMMENDATIONS ===
"""
                for i, rec in enumerate(recommendations[:3], 1):
                    summary_report += f"{i}. [{rec['priority']}] {rec['type']}: {rec['recommendation']}\n"
                
                st.download_button(
                    label="📝 Download Summary Report",
                    data=summary_report,
                    file_name=f"skillmirror_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
