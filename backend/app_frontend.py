import streamlit as st
import requests
import os
import re

# Constants
FASTAPI_URL = "http://localhost:8000"
MAX_FILE_SIZE_MB = 15

st.set_page_config(page_title="Farm360 Agent UI", page_icon="🚜", layout="centered")

# CSS for better chat UI
st.markdown("""
<style>
    .prediction-badge { padding: 5px 10px; border-radius: 5px; background-color: #2e7d32; color: white; font-weight: bold; }
    .chat-bubble { padding: 10px; border-radius: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

def check_backend_health():
    """Returns True if the FastAPI backend is responsive."""
    try:
        response = requests.get(f"{FASTAPI_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def parse_agent_response(text: str):
    """
    Parses the LLM's raw markdown text into specific semantic UI blocks.
    The LLM is prompted to include '1. Prediction:', '2. Explanation:', etc.
    If it fails to structure them rigidly, we gracefully display the raw text.
    """
    # Simple regex searching for the numbered lists or keyword headers
    sections = {
        "prediction": None,
        "explanation": None,
        "actions": None,
        "confidence": None,
        "raw": text
    }
    
    try:
        # Regex to capture content falling between numbered headers like "1. Prediction: ...."
        pred_match = re.search(r'(?:1\.\s*)?Prediction:\s*(.*?)(?=\n(?:2\.\s*)?Explanation:|\Z)', text, re.IGNORECASE | re.DOTALL)
        expl_match = re.search(r'(?:2\.\s*)?Explanation:\s*(.*?)(?=\n(?:3\.\s*)?Actionable Steps:|\Z)', text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r'(?:3\.\s*)?Actionable Steps:\s*(.*?)(?=\n(?:4\.\s*)?Confidence Level:|\Z)', text, re.IGNORECASE | re.DOTALL)
        conf_match = re.search(r'(?:4\.\s*)?Confidence Level:\s*(.*)', text, re.IGNORECASE | re.DOTALL)

        if pred_match: sections["prediction"] = pred_match.group(1).strip()
        if expl_match: sections["explanation"] = expl_match.group(1).strip()
        if action_match: sections["actions"] = action_match.group(1).strip()
        if conf_match: sections["confidence"] = conf_match.group(1).strip()
    except Exception:
        pass
        
    return sections

def render_response(sections: dict):
    if sections["prediction"] and sections["explanation"]:
        st.markdown(f"### ✨ Prediction")
        st.markdown(f"<div class='prediction-badge'>{sections['prediction']}</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown(f"### 📖 Explanation")
        st.info(sections["explanation"])
        
        if sections["actions"]:
            st.markdown(f"### 🛠 Actionable Steps")
            st.markdown(sections["actions"])
            
        if sections["confidence"]:
            st.markdown(f"### 📊 Confidence")
            conf = sections["confidence"].lower()
            if "high" in conf: st.success(sections["confidence"])
            elif "low" in conf: st.error(sections["confidence"])
            else: st.warning(sections["confidence"])
    else:
        # Fallback to standard raw render if LLM structure missed format
        st.markdown(sections["raw"])

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar (Configuration & Multimodal Inputs) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2821/2821814.png", width=100)
    st.title("Farm360 Setup")
    
    api_key = st.text_input("Enter FastAPI Secret Key:", type="password", value="default-secret-key")
    
    st.markdown("---")
    st.header("🖼️ Multimodal Analysis")
    uploaded_file = st.file_uploader("Upload Crop/Animal Image", type=["jpg", "png", "jpeg"])
    
    valid_file = None
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")
        else:
            st.image(uploaded_file, caption="Preview", use_container_width=True)
            valid_file = uploaded_file
            
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Interface ---
st.title("Farm360 Digital Assistant")

# Step 1: Health Check
is_healthy = check_backend_health()
if not is_healthy:
    st.error(f"🚨 Backend is unreachable at {FASTAPI_URL}. Start the FastAPI server first!")
    st.stop()

# Step 2: Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "structured" in msg:
            render_response(msg["structured"])
        else:
            st.markdown(msg["content"])
            if "image" in msg and msg["image"]:
                st.image(msg["image"], width=250)

# Step 3: Handle Input
user_query = st.chat_input("Ask Farm360 (e.g. 'Predict my rice yield' or 'Analyze this crop')...")
if user_query:
    # 1. Update UI instantly
    st.session_state.messages.append({"role": "user", "content": user_query, "image": valid_file})
    with st.chat_message("user"):
        st.markdown(user_query)
        if valid_file:
            st.image(valid_file, width=250)

    # 2. Setup networking Payload
    headers = {"X-API-Key": api_key}
    
    # 3. Processing Stream
    with st.chat_message("assistant"):
        with st.status("Routing through Farm360 Intelligence...", expanded=True) as status_box:
            st.write("Constructing payload...")
            
            try:
                if valid_file:
                    st.write("Uploading Image Tensor to model server...")
                    files = {"image": (valid_file.name, valid_file.getvalue(), valid_file.type)}
                    data = {"query": user_query}
                    
                    st.write("Calling multimodal inference endpoint...")
                    response = requests.post(f"{FASTAPI_URL}/analyze_image", headers=headers, files=files, data=data)
                else:
                    st.write("Calling text inference endpoint...")
                    response = requests.post(f"{FASTAPI_URL}/chat", headers=headers, data={"query": user_query})
                
                status_box.update(label="Decoding response...", state="running")
                response.raise_for_status()
                
                resp_json = response.json()
                raw_text = resp_json.get("response", "Error reading response")
                
                status_box.update(label="Complete!", state="complete", expanded=False)
                
                # 4. Render visually
                structured = parse_agent_response(raw_text)
                render_response(structured)
                
                # 5. Save to memory natively
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": raw_text,
                    "structured": structured
                })
                
            except requests.exceptions.HTTPError as http_err:
                status_box.update(label="Request Failed", state="error", expanded=True)
                if response.status_code == 401:
                    st.error("Authentication Failed: Invalid API Key.")
                else:
                    st.error(f"Internal Server Error: {response.text}")
                    
            except Exception as ex:
                status_box.update(label="System Error", state="error", expanded=True)
                st.error(f"An unexpected UI exception occurred: {str(ex)}")

