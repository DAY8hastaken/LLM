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
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Bot:")[-1].strip()
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
