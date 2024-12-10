import streamlit as st
from transformers import pipeline, set_seed
import torch

# Set page config
st.set_page_config(
    page_title="Story Continuation Generator",
    page_icon="📚",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    generator = pipeline('text-generation', model='gpt2')
    return generator

def generate_story_continuation(prompt, max_length=200, num_return_sequences=1):
    generator = load_model()
    set_seed(42)  # For reproducibility
    
    # Generate the continuation
    continuations = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=0.7,
        pad_token_id=50256
    )
    
    return [cont['generated_text'] for cont in continuations]

# UI Elements
st.title("📚 Story Continuation Generator")
st.markdown("""
Write the beginning of your story, and let AI help you continue it!
""")

# Input area
user_prompt = st.text_area("Enter the beginning of your story:", height=200,
                          placeholder="Once upon a time...")

# Sidebar options
with st.sidebar:
    st.header("Generation Settings")
    max_length = st.slider("Maximum length", 100, 500, 200)
    num_sequences = st.slider("Number of continuations", 1, 3, 1)

# Generate button
if st.button("Generate Continuation"):
    if user_prompt:
        with st.spinner("Generating story continuation..."):
            try:
                continuations = generate_story_continuation(
                    user_prompt,
                    max_length=max_length,
                    num_return_sequences=num_sequences
                )
                
                # Display continuations
                for i, continuation in enumerate(continuations, 1):
                    st.subheader(f"Continuation {i}")
                    # Format the continuation to remove the original prompt
                    formatted_continuation = continuation[len(user_prompt):].strip()
                    st.write(formatted_continuation)
                    st.markdown("---")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a story prompt first!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Hugging Face Transformers")
