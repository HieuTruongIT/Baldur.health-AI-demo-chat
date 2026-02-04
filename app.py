import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Baldur.health AI",
    page_icon="assets/logo.png",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stChatMessage {
        display: flex;
    }
    .stChatMessage.user {
        justify-content: flex-end;
    }
    .stChatMessage.user .stMarkdown {
        background-color: #DCF8C6;
        border-radius: 16px;
        padding: 10px 14px;
        max-width: 70%;
        text-align: left;
    }
    .stChatMessage.assistant {
        justify-content: flex-start;
    }
    .stChatMessage.assistant .stMarkdown {
        background-color: #F1F0F0;
        border-radius: 16px;
        padding: 10px 14px;
        max-width: 70%;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("assets/baldur.health.png", width=280)
st.markdown(
    "Baldur is focused on lifting healthcare outcomes for all through the adoption of new technologies."
)

MODEL_PATH = "Shinichii/baldur-health-ai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def build_prompt(messages):
    prompt = ""
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}\n"
    prompt += "Assistant: Answer:"
    return prompt

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Baldur Thinking..."):
            full_prompt = build_prompt(st.session_state.messages)

            inputs = tokenizer(
                full_prompt,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            match = re.search(
                r"answer:\s*(.*)",
                output_text,
                re.IGNORECASE | re.DOTALL
            )

            if match:
                output_text = match.group(1).strip()
            else:
                output_text = output_text.replace(full_prompt, "").strip()

            st.markdown(output_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": output_text
    })
