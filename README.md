# Local LLM Summarizer Comparer 🚀

An interactive Streamlit application to compare the performance of popular open-source summarization models (T5, DistilBART, PEGASUS) on user-provided text. This tool provides a clear, side-by-side analysis with key evaluation metrics to help users understand the strengths and weaknesses of each model.

---

## 1. Problem Definition, Domain Selection & Relevance

* **Problem Statement**: Evaluating and selecting the right text summarization model is a significant challenge for developers and researchers. Models vary in output style, accuracy, and computational cost. There's a need for an accessible tool that allows for direct, real-time comparison of different models on the same input text, complete with quantitative performance metrics.
* **Scope and Objectives**: This project aims to develop a self-contained web application that:
    1.  Allows a user to input any text.
    2.  Generates summaries using three distinct, popular transformer models: T5-Small, DistilBART, and PEGASUS.
    3.  Presents the generated summaries side-by-side for qualitative comparison.
    4.  Automatically calculates and displays quantitative metrics (Word Count, Readability, ROUGE-1 Recall) for each summary.
* **Relevance to LLMs**: This project directly addresses the practical application and evaluation of Large Language Models. It moves beyond theoretical performance by providing a hands-on environment to understand the nuanced differences in models fine-tuned for a specific NLP task (summarization), which is a core challenge in applied AI.

---

## 2. Literature Review & Background Study

This project is built upon foundational work in transformer architectures and text summarization.
* **T5 (Text-to-Text Transfer Transformer)**: Introduced by Raffel et al. (2020), T5 frames every NLP task as a text-to-text problem. Its versatility and strong performance on summarization benchmarks make it a crucial model for comparison. We use the `t5-small` variant as a lightweight but effective baseline.
* **DistilBART**: Sanh et al. (2019) introduced the concept of model distillation to create smaller, faster, and more efficient models. DistilBART is a distilled version of the BART model (Lewis et al., 2019), which is specifically designed for sequence-to-sequence tasks like summarization. It offers a balance between performance and computational efficiency.
* **PEGASUS**: Proposed by Zhang et al. (2020), PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive Summarization) uses a unique pre-training objective highly tailored for abstractive summarization. It often produces more fluent and human-like summaries, making it an excellent candidate for evaluating high-quality abstractive output.

This tool provides a practical means to observe the theoretical differences between these architectures in real-time.

---

## 3. Implementation & Technical Execution

The application is implemented as a unified Python script using the Streamlit framework, which handles both the frontend user interface and the backend model inference.
* **Core Framework**: Streamlit was chosen for its ability to rapidly create interactive data applications with simple Python scripts.
* **Model Integration**: The Hugging Face `transformers` library is used to access and run the pre-trained models. The `pipeline` function simplifies the inference process.
* **Efficiency**: To ensure a smooth user experience, model loading is a heavy operation. This is handled efficiently using Streamlit's `@st.cache_resource` decorator, which loads each model only once at the start of the session.
* **Metrics Calculation**: The `textstat` library is used for calculating the Flesch-Kincaid readability grade. Word Count and ROUGE-1 Recall are implemented with custom helper functions.
* **User Interface**: The UI is designed for clarity, with a large text area for input and a dynamically generated 3-column layout for displaying the results and their corresponding metrics.

---

## 4. Results Analysis

The tool was tested using a sample text about the James Webb Space Telescope. The results provide a clear example of the comparative insights this application can generate.

### Example Execution Results:

| Metric | T5-Small | DistilBART | PEGASUS |
| :--- | :--- | :--- | :--- |
| **Generated Summary** | *the James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy . its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble space Telescope . this has enabled a broad range of investigations across many fields of astronomie and cosmology .* | *The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy . Its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope . This has enabled a range of investigations across many fields of astronomy and cosmology .* | *The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy.<n>As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope.* |
| **Word Count** | 54 | 53 | 42 |
| **Readability Grade** | 12.1 | 12.6 | 14.0 |
| **ROUGE-1 Recall** | **0.667** | 0.650 | 0.567 |

### Analysis of Strengths and Weaknesses:
* **T5-Small** and **DistilBART** produced very similar, highly extractive summaries. They retained a large portion of the original text, resulting in high ROUGE-1 scores (0.667 and 0.650, respectively). This indicates they are effective at identifying and extracting key sentences. Their readability grade is around the 12th-grade level, similar to the source text.
* **PEGASUS** generated a more concise and abstractive summary. Its lower Word Count (42) and ROUGE-1 score (0.567) show that it paraphrased and condensed the information rather than just extracting it. This led to a slightly higher Readability Grade (14.0), suggesting more complex sentence construction.
* **Conclusion**: For tasks requiring factual extraction with minimal alteration, T5 and DistilBART are excellent choices. For tasks requiring a more human-like, condensed summary that captures the essence of the text, PEGASUS demonstrates superior abstractive capabilities. This tool successfully highlights these critical behavioral differences.
