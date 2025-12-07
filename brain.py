import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv # <--- Import this
from tools import find_doctor, get_medical_advice

# Load environment variables
load_dotenv()

# --- CACHED CLIENT ---
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY") # <--- Get from .env
    if not api_key:
        st.error("GROQ_API_KEY not found in .env file!")
        return None
    return Groq(api_key=api_key)

# ... (Rest of the file stays exactly the same)

# --- NEW ADVANCED PROMPT ---
# We tell the AI to ALWAYS identify the specialty, even if not asked.
SYSTEM_PROMPT = """
You are MedAgent-SL, a Sri Lankan Medical Triage Assistant.

Your Goal: Analyze symptoms and provide TWO things:
1. Search Query: Keywords to look up medical advice in the handbook.
2. Specialty: The Type of Specialist Doctor needed for this issue (e.g., Dermatologist, Cardiologist, Neurologist, Pediatrician, General Physician).

OUTPUT FORMAT:
Return valid JSON ONLY.
{
  "is_emergency": boolean,   // True if chest pain, heavy bleeding, difficulty breathing, fainting
  "search_query": "string",  // Keywords for the medical manual (e.g., "treatment for eczema")
  "specialty": "string",     // The relevant doctor type (e.g., "Dermatologist")
  "city": "string"           // Extract city if mentioned, else leave empty ""
}
"""

def process_case(user_text, visual_context=""):
    client = get_groq_client()
    full_query = f"User Complaint: {user_text}\nVisual Analysis: {visual_context}"
    
    try:
        # STEP 1: THINK (Analyze the case)
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_query}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(completion.choices[0].message.content)
        
        # STEP 2: SAFETY CHECK (Emergency)
        if analysis.get('is_emergency'):
            return "⚠️ **CRITICAL EMERGENCY DETECTED**\n\n**PLEASE HANG UP AND DIAL 1990 (SUWA SERIYA) IMMEDIATELY.**\nDo not rely on this app for life-threatening situations."

        # STEP 3: EXECUTE DUAL ACTIONS (Advice + Doctor)
        
        # Action A: Get Medical Advice (RAG)
        medical_facts = get_medical_advice(analysis['search_query'])
        
        # Action B: Get Doctor List (Database)
        # We automatically use the inferred specialty (e.g., "Dermatologist")
        doctor_list = find_doctor(analysis['specialty'], analysis.get('city', ''))

        # STEP 4: SYNTHESIZE FINAL RESPONSE
        # We ask the AI to combine the medical advice with the doctor list nicely.
        final_prompt = f"""
        You are a kind Sri Lankan doctor. Write a response to the patient.
        
        1. First, explain the condition and give advice based on these medical facts:
        "{medical_facts}"
        
        2. Then, tell the user you have found some relevant specialists in Sri Lanka for them.
        
        (Note: The doctor list will be added automatically after your message, so just introduce it).
        """
        
        final_response_gen = client.chat.completions.create(
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": full_query}
            ],
            model="llama-3.3-70b-versatile"
        )
        
        ai_message = final_response_gen.choices[0].message.content
        
        # Combine: AI Advice + The Hard Data Doctor List
        return f"{ai_message}\n\n---\n{doctor_list}"

    except Exception as e:
        return f"⚠️ System Error: {str(e)}"