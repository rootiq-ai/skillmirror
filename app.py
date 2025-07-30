import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter
import PyPDF2
import docx
import io
from datetime import datetime
import json

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

# Comprehensive skill categories
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
    
    # Check for resume length
    if len(resume_text) > 200:
        score += 15
    else:
        feedback.append("Resume seems too short - aim for 300+ words")
    
    # Check for section headers
    common_sections = ['experience', 'education', 'skills', 'summary', 'objective', 'work', 'employment']
    found_sections = sum(1 for section in common_sections if section in resume_text.lower())
    score += min(found_sections * 8, 25)
    
    if found_sections < 3:
        feedback.append("Add standard resume sections: Experience, Education, Skills")
    
    # Check for contact information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    if re.search(email_pattern, resume_text):
        score += 10
    else:
        feedback.append("Include your email address")
    
    if re.search(phone_pattern, resume_text):
        score += 10
    else:
        feedback.append("Include your phone number")
    
    # Check for dates (work experience)
    date_pattern = r'\b(19|20)\d{2}\b'
    if re.search(date_pattern, resume_text):
        score += 10
    else:
        feedback.append("Add dates to your work experience")
    
    # Check for action verbs
    action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'designed', 
                   'built', 'improved', 'increased', 'reduced', 'achieved', 'established',
                   'coordinated', 'executed', 'optimized', 'delivered']
    found_verbs = sum(1 for verb in action_verbs if verb in resume_text.lower())
    score += min(found_verbs * 2, 15)
    
    if found_verbs < 3:
        feedback.append("Use more action verbs (managed, developed, created, etc.)")
    
    # Check for quantifiable achievements
    number_pattern = r'\b\d+%?\b'
    numbers_found = len(re.findall(number_pattern, resume_text))
    score += min(numbers_found * 2, 15)
    
    if numbers_found < 3:
        feedback.append("Add quantified achievements (numbers, percentages, metrics)")
    
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
            'recommendation': f'Improve ATS score: {"; ".join(ats_feedback[:2])}'
        })
    
    # Skill gap recommendations
    if match_percentage < 60:
        top_gaps = []
        for category, gaps in skill_gaps.items():
            if gaps:
                top_gaps.extend(gaps[:2])
        
        if top_gaps:
            recommendations.append({
                'type': 'Skill Development',
                'priority': 'High',
                'recommendation': f'Learn key skills: {", ".join(top_gaps[:4])}'
            })
    
    # Additional recommendations
    recommendations.extend([
        {
            'type': 'Content Enhancement',
            'priority': 'Medium',
            'recommendation': 'Add quantified achievements with specific numbers and results'
        },
        {
            'type': 'Keywords',
            'priority': 'Medium',
            'recommendation': 'Include more industry keywords from the job posting'
        },
        {
            'type': 'Format',
            'priority': 'Low',
            'recommendation': 'Use consistent formatting and clear section headers'
        }
    ])
    
    return recommendations[:5]

# Main application
def main():
    st.markdown('<h1 class="main-header">🎯 SkillMirror</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Resume Analysis & Career Feedback</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload your resume** (PDF or DOCX)
        2. **Paste the job description**
        3. **Get instant AI feedback** including:
           - ATS compatibility score
           - Skill matching analysis
           - Personalized recommendations
        """)
        
        st.header("🎯 Analysis Features")
        st.markdown("""
        - **ATS Score** (0-100)
        - **Job Match** percentage
        - **Skill Gap** identification
        - **Visual Analytics**
        - **Export Results**
        """)
        
        st.info("💡 **Tip**: Use recent job postings for best results!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx'],
            help="Upload PDF or DOCX format"
        )
        
        resume_text = ""
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
            
            if resume_text:
                st.success(f"✅ Resume loaded! ({len(resume_text)} characters)")
                with st.expander("Preview Resume Text"):
                    st.text_area("Content Preview", resume_text[:500] + "..." if len(resume_text) > 500 else resume_text, height=150, disabled=True)
    
    with col2:
        st.header("💼 Job Description")
        job_description = st.text_area(
            "Paste the complete job description",
            height=300,
            placeholder="Paste the full job posting here, including requirements, responsibilities, and qualifications..."
        )
    
    # Analysis section
    if st.button("🔍 Analyze My Resume", type="primary", use_container_width=True):
        if not resume_text:
            st.error("❌ Please upload your resume first!")
        elif not job_description.strip():
            st.error("❌ Please paste the job description!")
        else:
            with st.spinner("🤖 Analyzing your resume..."):
                # Perform analysis
                resume_skills = extract_skills(resume_text, SKILL_CATEGORIES)
                ats_score, ats_feedback = calculate_ats_score(resume_text)
                skill_matches, skill_gaps, match_percentage = analyze_job_match(resume_skills, job_description)
                recommendations = generate_recommendations(ats_score, match_percentage, skill_gaps, ats_feedback)
            
            st.success("✅ Analysis Complete!")
            
            # Results display
            st.header("📊 Your Analysis Results")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score_color = "🟢" if ats_score >= 70 else "🟡" if ats_score >= 50 else "🔴"
                st.metric("ATS Score", f"{ats_score}/100", delta=f"{score_color}")
            
            with col2:
                match_color = "🟢" if match_percentage >= 60 else "🟡" if match_percentage >= 40 else "🔴"
                st.metric("Job Match", f"{match_percentage:.1f}%", delta=f"{match_color}")
            
            with col3:
                total_skills = sum(len(skills) for skills in resume_skills.values())
                st.metric("Skills Found", total_skills, delta="📝")
            
            # Visualizations
            st.header("📈 Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Skills match chart
                chart_data = []
                for category in SKILL_CATEGORIES.keys():
                    matches = len(skill_matches.get(category, []))
                    gaps = len(skill_gaps.get(category, []))
                    if matches > 0 or gaps > 0:
                        chart_data.append({'Category': category, 'Matches': matches, 'Gaps': gaps})
                
                if chart_data:
                    df = pd.DataFrame(chart_data)
                    fig = px.bar(
                        df.melt(id_vars=['Category'], var_name='Type', value_name='Count'),
                        x='Category', y='Count', color='Type',
                        title='Skills: Matches vs Gaps',
                        color_discrete_map={'Matches': '#4CAF50', 'Gaps': '#FF5722'}
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Score breakdown
                categories = ['Contact Info', 'Structure', 'Keywords', 'Achievements']
                scores = [20, 25, 30, 25]  # Simulated breakdown
                actual_scores = [min(s, ats_score/4) for s in scores]
                
                fig = go.Figure(data=[
                    go.Bar(x=categories, y=actual_scores, marker_color='#2196F3')
                ])
                fig.update_layout(title='ATS Score Breakdown', yaxis_title='Points')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed skill analysis
            st.header("🎯 Detailed Skill Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Skills You Have")
                for category, matches in skill_matches.items():
                    if matches:
                        st.write(f"**{category}:**")
                        for skill in matches[:5]:  # Show top 5
                            st.markdown(f'<span class="skill-match">{skill}</span>', unsafe_allow_html=True)
                        st.write("")
            
            with col2:
                st.subheader("❌ Skills You Need")
                for category, gaps in skill_gaps.items():
                    if gaps:
                        st.write(f"**{category}:**")
                        for skill in gaps[:5]:  # Show top 5
                            st.markdown(f'<span class="skill-gap">{skill}</span>', unsafe_allow_html=True)
                        st.write("")
            
            # Recommendations
            st.header("💡 Personalized Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                priority_icons = {'High': '🔥', 'Medium': '⚡', 'Low': '💡'}
                icon = priority_icons.get(rec['priority'], '📝')
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <h4>{icon} {rec['type']} ({rec['priority']} Priority)</h4>
                    <p>{rec['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ATS feedback
            if ats_feedback:
                st.header("⚠️ ATS Improvement Tips")
                for tip in ats_feedback[:3]:
                    st.markdown(f"""
                    <div class="warning-box">
                        <p>💡 {tip}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export results
            st.header("📥 Export Your Analysis")
            
            # Prepare export data
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'ats_score': ats_score,
                'match_percentage': round(match_percentage, 1),
                'total_skills_found': sum(len(skills) for skills in resume_skills.values()),
                'recommendations': recommendations,
                'top_skill_gaps': [skill for gaps in list(skill_gaps.values())[:3] for skill in gaps[:2]]
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                json_export = json.dumps(export_data, indent=2)
                st.download_button(
                    "📊 Download Analysis (JSON)",
                    json_export,
                    f"skillmirror_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    "application/json"
                )
            
            with col2:
                # Text summary
                summary = f"""
SKILLMIRROR ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

SUMMARY:
• ATS Score: {ats_score}/100
• Job Match: {match_percentage:.1f}%
• Skills Found: {sum(len(skills) for skills in resume_skills.values())}

TOP RECOMMENDATIONS:
"""
                for i, rec in enumerate(recommendations[:3], 1):
                    summary += f"{i}. {rec['type']}: {rec['recommendation']}\n"
                
                st.download_button(
                    "📝 Download Summary",
                    summary,
                    f"skillmirror_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    "text/plain"
                )

if __name__ == "__main__":
    main()
