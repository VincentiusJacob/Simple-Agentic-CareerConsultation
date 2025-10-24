import streamlit as st
from PIL import Image
import time

# web title 
st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🤖",
    layout="wide"
)

# set function for write stream input
welcome_sentence= "Welcome to AI Career Assistant! Unlock your career potential with our comprehensive AI-powered tools. Whether you're starting out, making a change, or advancing in your field, we've got you covered with personalized guidance and insights."
def stream():
    for word in welcome_sentence.split(" "):
        yield word + " "
        time.sleep(0.1)


st.title("AI Career Assistant")
st.write_stream(stream)

st.markdown("---")
st.subheader("Available Services :")

# standardize image size
size = (300, 200)

# Service 1
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader(" 🗣️ Career Q&A ")

    img1 = Image.open("qna.jpeg").resize(size)
    st.image(img1, width="stretch")

with col2:

    st.markdown("### Get instant answers to your career questions")
    st.markdown("- Job search strategies ")
    st.markdown("- Interview preparation")
    st.markdown("- Salary negotiation")
    st.markdown("- Career transitions")


# Service 2
col3, col4 = st.columns(2, gap="large")

with col3:
    st.subheader(" 📄 Resume Analyzer ")

    img2 = Image.open("resume_analyzer.jpeg").resize(size)
    st.image(img2, width="stretch")

with col4:

    st.markdown("### Optimize your resume for success")
    st.markdown("- AI-powered analysis")
    st.markdown("- Keyword optimization")
    st.markdown("- Format suggestions")
    st.markdown("- ATS compatibility check")


# Service 3
col5, col6 = st.columns(2, gap="large")
with col5:
    st.subheader(" 📊 Career Explorer")
    img3 = Image.open("career_exploration.jpeg").resize(size)
    st.image(img3, width="stretch")

with col6:
    
    st.markdown("### Discover your ideal career path")
    st.markdown("- Skill matching")
    st.markdown("- Industry insights")
    st.markdown("- Growth opportunities")
    st.markdown("- Salary expectations")

st.markdown("---")
st.subheader("Ready to Transform Your Career?")
st.write("Choose a service from the sidebar and let's get started on your journey to success!")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🎓")
    st.markdown("**Expert Guidance**")
    st.markdown("AI-trained on millions of career scenarios")

with col2:
    st.markdown("### ⚡")
    st.markdown("**Instant Results**")
    st.markdown("Get feedback and insights in seconds")

with col3:
    st.markdown("### 🔒")
    st.markdown("**Secure & Private**")
    st.markdown("Your data stays confidential")

with col4:
    st.markdown("### 🎯")
    st.markdown("**Personalized**")
    st.markdown("Tailored advice for your unique situation")
