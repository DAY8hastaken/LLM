import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- Load model ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    device = "cpu"   # Streamlit Cloud = CPU
    model.to(device)

    return tokenizer, model, device

tokenizer, model, device = load_model()

# --- Chat function ---
def chat(prompt):
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- UI ---
st.title("🤖 My AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_input = st.chat_input("Type something...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Generate response
    response = chat(user_input)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
