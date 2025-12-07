# app.py
import os
from dotenv import load_dotenv
load_dotenv() # <--- Load keys immediately
import streamlit as st
import asyncio
import base64
import edge_tts
from faster_whisper import WhisperModel
from groq import Groq
from brain import process_case

st.set_page_config(page_title="MedAgent SL", layout="wide", page_icon="🏥")
st.title("🏥 MedAgent SL: AI Triage System")

# --- CACHED WHISPER LOAD ---
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

whisper_model = load_whisper()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Patient Input")
    img_file = st.file_uploader("📷 Upload Photo (Skin/Wound)", type=['jpg', 'png'])
    audio_input = st.audio_input("🎤 Describe Symptoms") 

    if st.button("Consult Agent", type="primary"):
        if not audio_input and not img_file:
            st.error("Please upload an image or record audio.")
        else:
            with st.spinner("Analyzing Case..."):
                # A. VISION ANALYSIS (Fixed Model ID)
                visual_desc = ""
                if img_file:
                    bytes_data = img_file.getvalue()
                    b64 = base64.b64encode(bytes_data).decode()
                    
                    api_key = os.getenv("GROQ_API_KEY")
                    client = Groq(api_key=api_key)
                    
                    # UPDATED MODEL ID FOR 2025
                    v_resp = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct", # <--- NEW VISION MODEL
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this medical image clinically. Be concise."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                            ]
                        }]
                    )
                    visual_desc = v_resp.choices[0].message.content
                    
                    # SHOW VISION RESULT TO USER
                    st.success("✅ Image Analyzed")
                    st.markdown(f"**Visual Findings:** *{visual_desc}*")
                
                # B. AUDIO ANALYSIS
                user_text = "Analysis based on image only."
                if audio_input:
                    with open("temp.wav", "wb") as f:
                        f.write(audio_input.read())
                    segments, _ = whisper_model.transcribe("temp.wav")
                    user_text = " ".join([s.text for s in segments])
                    st.info(f"🗣️ **You said:** {user_text}")
                
                # C. BRAIN PROCESSING
                response = process_case(user_text, visual_desc)
                st.session_state.final_response = response

with col2:
    st.subheader("2. AI Diagnosis & Doctor Plan")
    if 'final_response' in st.session_state:
        # Display the text clearly
        st.markdown(st.session_state.final_response)
        
        # D. VOICE OUTPUT
        async def speak(text):
            # Clean asterisks for better speech
            clean_text = text.replace("*", "").replace("#", "")
            communicate = edge_tts.Communicate(clean_text, "en-US-AriaNeural")
            await communicate.save("output.mp3")
            
        asyncio.run(speak(st.session_state.final_response))
        st.audio("output.mp3", autoplay=True)