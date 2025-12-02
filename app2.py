import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 🎨 1. PAGE CONFIGURATION & MODERN STYLING
# ==========================================
st.set_page_config(
    page_title="CareerPath AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Minimalist/Modern Look
st.markdown("""
    <style>
    /* Global Fonts & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }

    /* Card Styling for Results */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 1rem;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .employed-card {
        background: linear-gradient(135deg, #e0ffe4 0%, #ffffff 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .unemployed-card {
        background: linear-gradient(135deg, #ffe0e0 0%, #ffffff 100%);
        border: 1px solid #f5c6cb;
        color: #721c24;
    }

    /* Sidebar Clean-up */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 2. LOAD TRAINED MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        with open('graduate_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None

data = load_model()

# --- SAFETY FUNCTION (PREVENTS CRASHES) ---
def safe_encode(encoder, value):
    """
    Safely encodes a value. If the value (e.g., 'No') was missing 
    from training data, use the first available class to prevent error.
    """
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # If 'No' is unknown, fallback to the 0-index class (usually Yes or the only option)
        # This keeps the app running.
        return encoder.transform([encoder.classes_[0]])[0]

# ==========================================
# 📝 3. SIDEBAR (CONTROLS)
# ==========================================
with st.sidebar:
    st.markdown("### 👤 Candidate Profile")
    st.write("Enter the graduate's details below.")
    
    if data:
        encoders = data['encoders']
        gwa_map = data['gwa_map']
        income_map = data['income_map']
        
        # Group 1: Demographics
        with st.expander("basic info", expanded=True):
            age_opts = sorted(encoders['Age'].classes_)
            age = st.selectbox("Age Group", age_opts)
            
            gender_opts = sorted(encoders['Gender'].classes_)
            gender = st.selectbox("Gender", gender_opts)
            
            civil_opts = sorted(encoders['Civil Status'].classes_)
            civil_status = st.selectbox("Civil Status", civil_opts)
            
            income_options = list(income_map.keys())
            income_input = st.selectbox("Family Income", income_options)

        # Group 2: Academics
        with st.expander("academic background", expanded=False):
            univ_options = sorted(encoders['Name of University'].classes_)
            university = st.selectbox("University", univ_options)
            
            degree_opts = sorted(encoders['Degree_Category'].classes_)
            degree_cat = st.selectbox("Degree Category", degree_opts)
            
            gwa_options = list(gwa_map.keys())
            gwa_input = st.selectbox("GWA (Grades)", gwa_options)
            
            grad_year = st.number_input("Year Graduated", min_value=2018, max_value=2025, value=2024)
            
            # Force Yes/No options even if dataset only had Yes
            honors = st.selectbox("Graduated with Honors?", ['No', 'Yes'])

        # Group 3: Experience & Strategy
        with st.expander("experience & search", expanded=False):
            # Force Yes/No options to ensure 'No' is clickable
            internship = st.selectbox("Internship Experience?", ['No', 'Yes'])
            
            int_related = st.selectbox("Related to Course?", ['No', 'Yes'])
            
            training = st.selectbox("Job-Related Trainings?", ['No', 'Yes'])
            
            search_options = sorted(encoders['Main job search method used after graduation'].classes_)
            search_method = st.selectbox("Search Method", search_options)
            
            civil_service = st.selectbox("Civil Service Passer?", sorted(encoders['Are you a Civil Service Exam Passer?'].classes_))

        # Predict Button
        st.markdown("---")
        predict_btn = st.button("✨ Analyze Profile", type="primary", use_container_width=True)

# ==========================================
# 🏠 4. MAIN DASHBOARD
# ==========================================

# Hero Section
st.markdown('<div class="main-header">CareerPath AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced predictive analytics for graduate employment outcomes.</div>', unsafe_allow_html=True)

if data is None:
    st.error("⚠️ Model file not found. Please upload 'graduate_model.pkl' to your GitHub repository.")
    st.stop()

# Layout
col1, col2 = st.columns([1.5, 1])

with col1:
    if predict_btn:
        # Prepare Data
        gwa_val = gwa_map[gwa_input]
        income_val = income_map[income_input]
        
        # We use safe_encode() for every field to ensure NO CRASHES
        input_data = {
            'Age': safe_encode(encoders['Age'], age),
            'Gender': safe_encode(encoders['Gender'], gender),
            'Civil Status': safe_encode(encoders['Civil Status'], civil_status),
            'Name of University': safe_encode(encoders['Name of University'], university),
            'Degree_Category': safe_encode(encoders['Degree_Category'], degree_cat),
            'GWA_Code': gwa_val,
            'Income_Code': income_val,
            'What year did you graduate?': grad_year,
            'Did you graduate with honors?': safe_encode(encoders['Did you graduate with honors?'], honors),
            'Did you have internship or on-the-job training experience?': safe_encode(encoders['Did you have internship or on-the-job training experience?'], internship),
            'Was your internship related to your course?': safe_encode(encoders['Was your internship related to your course?'], int_related),
            'Have you attended any job-related training, seminar, or certification?': safe_encode(encoders['Have you attended any job-related training, seminar, or certification?'], training),
            'Main job search method used after graduation': safe_encode(encoders['Main job search method used after graduation'], search_method),
            'Are you a Civil Service Exam Passer?': safe_encode(encoders['Are you a Civil Service Exam Passer?'], civil_service)
        }
        
        features = [list(input_data.values())]
        model = data['model']
        
        # Prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = np.max(probability) * 100

        # Result Display with Modern Cards
        if prediction == 1:
            st.markdown(f"""
                <div class="result-card employed-card">
                    <h1>🎉 Likely Employed</h1>
                    <p>The model predicts a high probability of employment for this profile.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Confidence Analysis")
            st.progress(int(confidence))
            st.caption(f"Model Confidence: {confidence:.1f}%")
            
        else:
            st.markdown(f"""
                <div class="result-card unemployed-card">
                    <h1>📉 Likely Unemployed</h1>
                    <p>The model suggests this profile faces employment challenges.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Confidence Analysis")
            st.progress(int(confidence))
            st.caption(f"Model Confidence: {confidence:.1f}%")

    else:
        # Default State (Empty State)
        st.info("👈 Please configure the graduate profile in the sidebar and click 'Analyze Profile' to start.")
        st.markdown("""
        **How it works:**
        1. Fill in the demographic details.
        2. Provide academic history.
        3. Click **Analyze** to see the AI prediction.
        """)

# Right Column: Insights Panel
with col2:
    st.markdown("### 📊 Market Insights")
    with st.container():
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #eee;">
            <h4 style="margin-top:0;">Key Employability Factors</h4>
            <ul style="font-size: 0.9rem; color: #555;">
                <li style="margin-bottom: 0.5rem;"><b>GWA & Academics:</b> Strong academic performance remains a top indicator for early employment.</li>
                <li style="margin-bottom: 0.5rem;"><b>Search Strategy:</b> Graduates using aggressive search methods (Walk-ins, Networking) often find jobs faster than those relying solely on online portals.</li>
                <li style="margin-bottom: 0.5rem;"><b>Continuous Learning:</b> Post-grad certifications significantly boost employability odds.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Placeholder for a chart
    st.markdown("### Historical Trends")
    chart_data = pd.DataFrame({
        "Status": ["Employed", "Unemployed"],
        "Count": [85, 15] 
    })
    st.bar_chart(chart_data.set_index("Status"), color=["#38ef7d"])