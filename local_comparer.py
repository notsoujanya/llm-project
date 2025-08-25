import streamlit as st
import textstat
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Summarizer Comparer 🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Model Loading (Cached for performance) ---
# Using @st.cache_resource to load models only once
@st.cache_resource
def load_summarizer_models():
    """Loads and caches the Hugging Face summarization models."""
    with st.spinner("Loading models... This may take a few minutes on first run."):
        t5_summarizer = pipeline("summarization", model="t5-small")
        bart_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        pegasus_summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
    return {
        "T5-Small": t5_summarizer,
        "DistilBART": bart_summarizer,
        "PEGASUS": pegasus_summarizer,
    }

# --- Helper Functions for Metrics ---
def calculate_metrics(generated_text, source_text):
    """Calculates all evaluation metrics for a given text."""
    if not generated_text or not isinstance(generated_text, str):
        return {"word_count": 0, "readability_grade": 0, "rouge1_recall": 0}

    # 1. Word Count
    word_count = len(generated_text.split())

    # 2. Readability (Flesch-Kincaid Grade)
    try:
        readability_grade = textstat.flesch_kincaid_grade(generated_text)
    except:
        readability_grade = 0 # Handle cases with too little text

    # 3. ROUGE-1 Recall
    source_words = set(source_text.lower().split())
    generated_words = generated_text.lower().split()
    if not source_words:
        rouge1_recall = 0
    else:
        overlapping_words = [w for w in source_words if w in generated_words]
        rouge1_recall = len(overlapping_words) / len(source_words)

    return {
        "word_count": word_count,
        "readability_grade": readability_grade,
        "rouge1_recall": rouge1_recall,
    }

# --- UI Rendering ---

# Header
st.title("Local LLM Summarizer Comparer 🚀")
st.markdown("An Interactive Tool to Compare Open-Source Summarization Models")
st.divider()

# Load models at the start
summarizer_models = load_summarizer_models()
st.success("Models loaded successfully!")

# Documentation Section
with st.expander("📘 Project Documentation", expanded=False):
    st.markdown("""
    ### 1. Problem Definition
    **Problem:** Different text summarization models have unique strengths and weaknesses. Choosing the right model for a specific task requires a direct comparison of their outputs and performance.
    **Objective:** This Streamlit application provides an easy-to-use interface for comparing the outputs of several popular, locally-run summarization models (T5, DistilBART, PEGASUS) and evaluating them with quantitative metrics.

    ### 2. Model Selection
    - **T5 (Text-to-Text Transfer Transformer):** A versatile model by Google AI that frames all NLP tasks as a text-to-text problem, making it excellent for summarization.
    - **DistilBART:** A distilled, lighter, and faster version of the BART model, specifically fine-tuned for high-quality text summarization.
    - **PEGASUS:** A model from Google whose pre-training objective is specifically tailored for abstractive summarization, often yielding highly coherent and human-like summaries.

    ### 3. Implementation (Streamlit Edition)
    - **Unified Architecture:** This app is a self-contained Python script using the Streamlit framework. Streamlit handles both the frontend UI and the backend logic.
    - **Local Inference:** All models are downloaded from Hugging Face and run on your local machine (or the server hosting the app). No APIs are needed.
    - **Model Caching:** Models are loaded once and cached using Streamlit's `@st.cache_resource` decorator for efficient performance.

    ### 4. Evaluation Metrics
    - **Word Count:** Basic metric for controlling output verbosity.
    - **Readability (Flesch-Kincaid Grade):** Estimates the US school-grade level required to understand the text.
    - **ROUGE-1 Recall:** Measures the overlap of individual words between the generated output and the original source text.

    ### 5. How to Execute
    1.  Save this code as a Python file (e.g., `local_comparer.py`).
    2.  Create a `requirements.txt` file with the necessary libraries.
    3.  Run `pip install -r requirements.txt` in your terminal.
    4.  Run `streamlit run local_comparer.py` to launch the application.
    """)

# Main Tool Section
st.header("✍️ Input Text for Summarization")

source_text_input = st.text_area(
    "**Source Text**",
    height=300,
    value="The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope. This has enabled a broad range of investigations across many fields of astronomy and cosmology, such as observation of the first stars and the formation of the first galaxies, and detailed atmospheric characterization of potentially habitable exoplanets."
)

# Generate Button
st.divider()
if st.button("📊 Generate & Compare Summaries", type="primary", use_container_width=True):
    if not source_text_input.strip():
        st.error("Please provide some source text to summarize.")
    else:
        with st.spinner("🧠 Models are summarizing... Please wait."):
            # --- Model Inference ---
            results = {}
            for model_name, summarizer in summarizer_models.items():
                try:
                    # Summarizers expect a list of texts
                    output = summarizer(source_text_input, max_length=150, min_length=30, do_sample=False)
                    results[model_name] = output[0]['summary_text']
                except Exception as e:
                    results[model_name] = f"Error processing with {model_name}: {e}"

        # --- Display Results ---
        st.header("🔍 Comparative Analysis")

        cols = st.columns(len(results))

        for i, (model_name, generated_text) in enumerate(results.items()):
            with cols[i]:
                st.subheader(model_name)
                metrics = calculate_metrics(generated_text, source_text_input)
                with st.container(border=True):
                    st.markdown(generated_text)
                    st.divider()
                    st.markdown(f"""
                    - **Word Count:** `{metrics['word_count']}`
                    - **Readability Grade:** `{metrics['readability_grade']:.1f}`
                    - **ROUGE-1 Recall:** `{metrics['rouge1_recall']:.3f}`
                    """)