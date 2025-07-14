import streamlit as st
import json
import re
import io
import PyPDF2
import docx
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
import zipfile
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Bulk Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class ResumeData:
    filename: str
    personal_info: Dict
    summary: str
    skills: List[str]
    experience: List[Dict]
    education: List[Dict]
    certifications: List[str]
    raw_text: str
    match_score: float = 0.0
    matched_skills: List[str] = None
    missing_skills: List[str] = None

class BulkResumeParser:
    def __init__(self):
        self.skills_keywords = [
            'python', 'java', 'javascript', 'react', 'node.js', 'django', 'flask',
            'aws', 'docker', 'kubernetes', 'git', 'sql', 'mongodb', 'postgresql',
            'machine learning', 'data science', 'artificial intelligence', 'deep learning',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
            'html', 'css', 'angular', 'vue.js', 'typescript', 'c++', 'c#', 'go',
            'rust', 'php', 'ruby', 'swift', 'kotlin', 'flutter', 'dart', 'scala',
            'agile', 'scrum', 'devops', 'ci/cd', 'jenkins', 'github', 'gitlab',
            'microservices', 'rest api', 'graphql', 'redis', 'elasticsearch',
            'spark', 'hadoop', 'kafka', 'rabbitmq', 'nginx', 'apache', 'linux',
            'spring', 'hibernate', 'maven', 'gradle', 'junit', 'selenium', 'jest',
            'express', 'fastapi', 'laravel', 'rails', 'django rest framework'
        ]
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file content"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file content"""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from TXT file content"""
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    
    def extract_email(self, text: str) -> str:
        """Extract email from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else "Email not found"
    
    def extract_phone(self, text: str) -> str:
        """Extract phone number from text"""
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        return ''.join(phones[0]) if phones else "Phone not found"
    
    def extract_name(self, text: str) -> str:
        """Extract name from text"""
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 4 and not '@' in line and not any(char.isdigit() for char in line):
                return line
        return "Name not found"
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skills_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return list(set(found_skills))
    
    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience"""
        experience = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for job titles or company patterns
            if any(word in line.lower() for word in ['engineer', 'developer', 'analyst', 'manager', 'consultant']):
                if len(experience) < 3:
                    experience.append({
                        "position": line[:50],
                        "company": "Company Name",
                        "duration": "2020-Present",
                        "description": line
                    })
        
        if not experience:
            experience = [{
                "position": "Previous Position",
                "company": "Previous Company",
                "duration": "2020-Present",
                "description": "Work experience details"
            }]
        
        return experience
    
    def extract_education(self, text: str) -> List[Dict]:
        """Extract education information"""
        education = []
        edu_keywords = ['bachelor', 'master', 'phd', 'university', 'college', 'degree']
        
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in edu_keywords):
                education.append({
                    "degree": line.strip()[:50],
                    "institution": "University Name",
                    "year": "2020"
                })
                break
        
        if not education:
            education = [{"degree": "Bachelor's Degree", "institution": "University", "year": "2020"}]
        
        return education
    
    def extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        cert_keywords = ['certified', 'certification', 'certificate']
        certifications = []
        
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in cert_keywords):
                certifications.append(line.strip()[:50])
        
        return certifications[:5]
    
    def parse_resume_content(self, filename: str, file_content: bytes) -> ResumeData:
        """Parse resume content and extract structured data"""
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            text = self.extract_text_from_pdf(file_content)
        elif file_ext == 'docx':
            text = self.extract_text_from_docx(file_content)
        elif file_ext == 'txt':
            text = self.extract_text_from_txt(file_content)
        else:
            text = "Unsupported file type"
        
        personal_info = {
            "name": self.extract_name(text),
            "email": self.extract_email(text),
            "phone": self.extract_phone(text),
            "location": "Location not specified"
        }
        
        return ResumeData(
            filename=filename,
            personal_info=personal_info,
            summary=text[:200] + "..." if len(text) > 200 else text,
            skills=self.extract_skills(text),
            experience=self.extract_experience(text),
            education=self.extract_education(text),
            certifications=self.extract_certifications(text),
            raw_text=text
        )

class JDSkillExtractor:
    def __init__(self):
        self.common_skills = [
            'Python', 'Java', 'JavaScript', 'React', 'Node.js', 'Django', 'Flask',
            'AWS', 'Docker', 'Kubernetes', 'Git', 'SQL', 'MongoDB', 'PostgreSQL',
            'Machine Learning', 'Data Science', 'AI', 'Deep Learning', 'TensorFlow',
            'PyTorch', 'HTML', 'CSS', 'Angular', 'Vue.js', 'TypeScript', 'C++',
            'C#', 'Go', 'Rust', 'PHP', 'Ruby', 'Swift', 'Kotlin', 'Flutter',
            'Agile', 'Scrum', 'DevOps', 'CI/CD', 'Jenkins', 'GitHub', 'GitLab',
            'Microservices', 'REST API', 'GraphQL', 'Redis', 'Elasticsearch'
        ]
    
    def extract_skills_from_jd(self, job_description: str) -> List[str]:
        """Extract skills from job description"""
        jd_lower = job_description.lower()
        extracted_skills = []
        
        for skill in self.common_skills:
            if skill.lower() in jd_lower:
                extracted_skills.append(skill)
        
        # Additional pattern matching for common terms
        patterns = [
            r'\b\d+\+?\s*years?\s*(?:of\s*)?(?:experience\s*)?(?:in\s*)?([A-Za-z\s\.\-]+)',
            r'(?:experience\s*(?:with|in)\s*)([A-Za-z\s\.\-]+)',
            r'(?:knowledge\s*(?:of|in)\s*)([A-Za-z\s\.\-]+)',
            r'(?:proficient\s*(?:in|with)\s*)([A-Za-z\s\.\-]+)',
            r'(?:skilled\s*(?:in|with)\s*)([A-Za-z\s\.\-]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            for match in matches:
                skill = match.strip().title()
                if len(skill) > 2 and len(skill) < 50 and skill not in extracted_skills:
                    extracted_skills.append(skill)
        
        return list(set(extracted_skills))[:15]  # Limit to 15 skills

class ResumeAnalyzer:
    def __init__(self, api_key: str = None):
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
    
    def calculate_match_score(self, resume: ResumeData, required_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        """Calculate match score between resume and required skills"""
        resume_skills_lower = [skill.lower() for skill in resume.skills]
        required_skills_lower = [skill.lower() for skill in required_skills]
        
        matched_skills = []
        missing_skills = []
        
        for req_skill in required_skills:
            req_skill_lower = req_skill.lower()
            found = False
            
            for resume_skill in resume_skills_lower:
                if req_skill_lower in resume_skill or resume_skill in req_skill_lower:
                    matched_skills.append(req_skill)
                    found = True
                    break
            
            if not found:
                missing_skills.append(req_skill)
        
        # Calculate score
        if len(required_skills) > 0:
            match_score = (len(matched_skills) / len(required_skills)) * 100
        else:
            match_score = 0
        
        # Boost score for additional relevant skills
        additional_score = min(20, len(resume.skills) * 2)
        match_score = min(100, match_score + additional_score)
        
        return match_score, matched_skills, missing_skills
    
    def analyze_bulk_resumes(self, resumes: List[ResumeData], required_skills: List[str]) -> List[ResumeData]:
        """Analyze multiple resumes and rank them"""
        analyzed_resumes = []
        
        for resume in resumes:
            match_score, matched_skills, missing_skills = self.calculate_match_score(resume, required_skills)
            
            resume.match_score = match_score
            resume.matched_skills = matched_skills
            resume.missing_skills = missing_skills
            
            analyzed_resumes.append(resume)
        
        # Sort by match score (descending)
        analyzed_resumes.sort(key=lambda x: x.match_score, reverse=True)
        
        return analyzed_resumes

def main():
    st.title("🎯 Bulk Resume Analyzer")
    st.write("Upload multiple resumes, analyze against job requirements, and find the best matches")
    
    # Sidebar
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
    
    # Initialize components
    parser = BulkResumeParser()
    extractor = JDSkillExtractor()
    analyzer = ResumeAnalyzer(api_key)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Bulk Upload", "🔍 Job Analysis", "🏆 Rankings", "📊 Export"])
    
    with tab1:
        st.header("Upload Multiple Resumes")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload options
            upload_option = st.radio("Choose upload method:", ["Individual Files", "ZIP Archive"])
            
            if upload_option == "Individual Files":
                uploaded_files = st.file_uploader(
                    "Upload resume files",
                    type=['pdf', 'docx', 'txt'],
                    accept_multiple_files=True,
                    help="Select multiple resume files"
                )
                
                if uploaded_files:
                    with st.spinner(f"Processing {len(uploaded_files)} resumes..."):
                        resumes = []
                        
                        progress_bar = st.progress(0)
                        for i, file in enumerate(uploaded_files):
                            resume_data = parser.parse_resume_content(file.name, file.read())
                            resumes.append(resume_data)
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        st.session_state.resumes = resumes
                        st.success(f"✅ Successfully processed {len(resumes)} resumes!")
            
            else:  # ZIP Archive
                zip_file = st.file_uploader(
                    "Upload ZIP file containing resumes",
                    type=['zip'],
                    help="ZIP file should contain PDF, DOCX, or TXT files"
                )
                
                if zip_file:
                    with st.spinner("Extracting and processing resumes from ZIP..."):
                        resumes = []
                        
                        with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zip_ref:
                            file_list = [f for f in zip_ref.namelist() if f.endswith(('.pdf', '.docx', '.txt'))]
                            
                            progress_bar = st.progress(0)
                            for i, filename in enumerate(file_list):
                                try:
                                    file_content = zip_ref.read(filename)
                                    resume_data = parser.parse_resume_content(filename, file_content)
                                    resumes.append(resume_data)
                                except Exception as e:
                                    st.warning(f"Could not process {filename}: {str(e)}")
                                
                                progress_bar.progress((i + 1) / len(file_list))
                        
                        st.session_state.resumes = resumes
                        st.success(f"✅ Successfully processed {len(resumes)} resumes from ZIP!")
        
        with col2:
            if 'resumes' in st.session_state:
                st.subheader("📊 Upload Summary")
                
                resumes = st.session_state.resumes
                
                # Statistics
                total_resumes = len(resumes)
                total_skills = sum(len(r.skills) for r in resumes)
                avg_skills = total_skills / total_resumes if total_resumes > 0 else 0
                
                st.metric("Total Resumes", total_resumes)
                st.metric("Total Skills Found", total_skills)
                st.metric("Avg Skills per Resume", f"{avg_skills:.1f}")
                
                # Top skills across all resumes
                all_skills = []
                for resume in resumes:
                    all_skills.extend(resume.skills)
                
                if all_skills:
                    skill_counts = {}
                    for skill in all_skills:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
                    
                    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    st.subheader("🔝 Top Skills Found")
                    for skill, count in top_skills:
                        st.write(f"{skill}: {count} resumes")
    
    with tab2:
        st.header("Job Description & Skill Analysis")
        
        if 'resumes' not in st.session_state:
            st.warning("Please upload resumes first!")
            return
        
        # Job description input
        job_description = st.text_area(
            "Enter Job Description",
            height=200,
            placeholder="Paste the complete job description here...",
            help="Enter the job description to extract required skills"
        )
        
        if job_description:
            # Extract skills from JD
            with st.spinner("Extracting skills from job description..."):
                extracted_skills = extractor.extract_skills_from_jd(job_description)
            
            st.subheader("📋 Skills Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Auto-Extracted Skills")
                st.write("*Skills automatically detected from job description*")
                
                if extracted_skills:
                    st.write("**Extracted Skills:**")
                    for skill in extracted_skills:
                        st.write(f"• {skill}")
                    
                    # Allow editing of extracted skills
                    edited_extracted = st.text_area(
                        "Edit extracted skills (one per line):",
                        value="\n".join(extracted_skills),
                        height=150,
                        key="extracted_skills"
                    )
                    extracted_skills = [skill.strip() for skill in edited_extracted.split('\n') if skill.strip()]
                else:
                    st.info("No skills automatically detected. Please add skills manually.")
                    extracted_skills = []
            
            with col2:
                st.subheader("✏️ Manual Skills")
                st.write("*Add additional skills manually*")
                
                manual_skills_input = st.text_area(
                    "Add manual skills (one per line):",
                    placeholder="React\nPython\nAWS\nDocker\nAgile",
                    height=200,
                    key="manual_skills"
                )
                
                manual_skills = [skill.strip() for skill in manual_skills_input.split('\n') if skill.strip()]
                
                if manual_skills:
                    st.write("**Manual Skills:**")
                    for skill in manual_skills:
                        st.write(f"• {skill}")
            
            # Combine all skills
            all_required_skills = list(set(extracted_skills + manual_skills))
            
            if all_required_skills:
                st.subheader("🎯 Combined Required Skills")
                st.write(f"**Total: {len(all_required_skills)} skills**")
                
                for skill in all_required_skills:
                    st.write(f"• {skill}")
                
                # Analyze button
                if st.button("🔍 Analyze All Resumes", type="primary"):
                    with st.spinner("Analyzing all resumes against requirements..."):
                        analyzed_resumes = analyzer.analyze_bulk_resumes(
                            st.session_state.resumes, 
                            all_required_skills
                        )
                        st.session_state.analyzed_resumes = analyzed_resumes
                        st.session_state.required_skills = all_required_skills
                        st.success("✅ Analysis completed!")
                        st.rerun()
    
    with tab3:
        st.header("🏆 Resume Rankings & Results")
        
        if 'analyzed_resumes' not in st.session_state:
            st.warning("Please complete the analysis first!")
            return
        
        analyzed_resumes = st.session_state.analyzed_resumes
        required_skills = st.session_state.required_skills
        
        # Summary statistics
        st.subheader("📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resumes", len(analyzed_resumes))
        
        with col2:
            avg_score = np.mean([r.match_score for r in analyzed_resumes])
            st.metric("Average Match", f"{avg_score:.1f}%")
        
        with col3:
            top_matches = len([r for r in analyzed_resumes if r.match_score >= 70])
            st.metric("Top Matches (70%+)", top_matches)
        
        with col4:
            good_matches = len([r for r in analyzed_resumes if r.match_score >= 50])
            st.metric("Good Matches (50%+)", good_matches)
        
        # Score distribution chart
        st.subheader("📈 Score Distribution")
        
        scores = [r.match_score for r in analyzed_resumes]
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="Resume Match Score Distribution",
            labels={'x': 'Match Score (%)', 'y': 'Number of Resumes'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed rankings
        st.subheader("🎯 Detailed Rankings")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            min_score = st.slider("Minimum Match Score", 0, 100, 0, 5)
        
        with col2:
            show_top_n = st.selectbox("Show top N results", [5, 10, 20, 50, 100], index=1)
        
        # Filter and limit results
        filtered_resumes = [r for r in analyzed_resumes if r.match_score >= min_score][:show_top_n]
        
        for i, resume in enumerate(filtered_resumes):
            st.subheader(f"#{i+1} - {resume.personal_info['name']} ({resume.match_score:.1f}%)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**📧 Email:** {resume.personal_info['email']}")
                st.write(f"**📱 Phone:** {resume.personal_info['phone']}")
                st.write(f"**📄 File:** {resume.filename}")
            
            with col2:
                st.write(f"**🎯 Match Score:** {resume.match_score:.1f}%")
                st.write(f"**✅ Matched Skills:** {len(resume.matched_skills)}")
                st.write(f"**❌ Missing Skills:** {len(resume.missing_skills)}")
            
            # Expandable details
            with st.expander(f"View Details - {resume.personal_info['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("✅ Matched Skills")
                    if resume.matched_skills:
                        for skill in resume.matched_skills:
                            st.write(f"• {skill}")
                    else:
                        st.info("No matched skills found")
                    
                    st.subheader("📚 All Resume Skills")
                    if resume.skills:
                        for skill in resume.skills:
                            st.write(f"• {skill}")
                    else:
                        st.info("No skills found")
                
                with col2:
                    st.subheader("❌ Missing Skills")
                    if resume.missing_skills:
                        for skill in resume.missing_skills:
                            st.write(f"• {skill}")
                    else:
                        st.info("No missing skills")
                    
                    st.subheader("💼 Experience")
                    for exp in resume.experience:
                        st.write(f"**{exp['position']}** at {exp['company']}")
                        st.write(f"Duration: {exp['duration']}")
                        st.write(f"Description: {exp['description']}")
                        st.write("---")
                    
                    st.subheader("🎓 Education")
                    for edu in resume.education:
                        st.write(f"**{edu['degree']}** - {edu['institution']} ({edu['year']})")
                    
                    if resume.certifications:
                        st.subheader("📜 Certifications")
                        for cert in resume.certifications:
                            st.write(f"• {cert}")
    
    with tab4:
        st.header("📊 Export Results")
        
        if 'analyzed_resumes' not in st.session_state:
            st.warning("Please complete the analysis first!")
            return
        
        analyzed_resumes = st.session_state.analyzed_resumes
        
        # Export options
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
        
        with col2:
            include_details = st.checkbox("Include detailed information", value=True)
        
        if st.button("Generate Export", type="primary"):
            with st.spinner("Preparing export..."):
                # Prepare data for export
                export_data = []
                
                for i, resume in enumerate(analyzed_resumes):
                    row = {
                        'Rank': i + 1,
                        'Name': resume.personal_info['name'],
                        'Email': resume.personal_info['email'],
                        'Phone': resume.personal_info['phone'],
                        'Filename': resume.filename,
                        'Match Score': resume.match_score,
                        'Matched Skills Count': len(resume.matched_skills),
                        'Missing Skills Count': len(resume.missing_skills),
                        'Total Skills': len(resume.skills)
                    }
                    
                    if include_details:
                        row.update({
                            'Matched Skills': ', '.join(resume.matched_skills),
                            'Missing Skills': ', '.join(resume.missing_skills),
                            'All Skills': ', '.join(resume.skills),
                            'Summary': resume.summary
                        })
                    
                    export_data.append(row)
                
                # Create DataFrame
                df = pd.DataFrame(export_data)
                
                if export_format == "CSV":
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                elif export_format == "JSON":
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                elif export_format == "Excel":
                    buffer = io.BytesIO()
                    df.to_excel(buffer, index=False)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.success("Export ready!")
        
        # Preview data
        if st.checkbox("Preview Export Data"):
            if 'analyzed_resumes' in st.session_state:
                preview_data = []
                for i, resume in enumerate(analyzed_resumes[:5]):  # Show first 5
                    preview_data.append({
                        'Rank': i + 1,
                        'Name': resume.personal_info['name'],
                        'Email': resume.personal_info['email'],
                        'Match Score': f"{resume.match_score:.1f}%",
                        'Matched Skills': len(resume.matched_skills),
                        'Missing Skills': len(resume.missing_skills)
                    })
                
                st.dataframe(pd.DataFrame(preview_data))

if __name__ == "__main__":
    main()