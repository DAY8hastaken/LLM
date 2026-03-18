import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# --- Load model ---
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device

tokenizer, model, device = load_model()

# --- Chat function ---
def chat(prompt):
    text = f"User: {prompt}\nBot:"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
