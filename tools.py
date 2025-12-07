import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. DOCTOR DATABASE (Structured Data)
# ==========================================
SRI_LANKAN_DOCTORS = [
    {"name": "Dr. S. Gunawardena", "specialty": "Dermatologist", "hospital": "Asiri Surgical", "city": "Colombo", "contact": "011-4523300"},
    {"name": "Dr. A. Pathirana", "specialty": "Dermatologist", "hospital": "Durdans Hospital", "city": "Colombo", "contact": "011-2140000"},
    {"name": "Dr. Amal Perera", "specialty": "Cardiologist", "hospital": "National Hospital", "city": "Colombo", "contact": "011-2691111"},
    {"name": "Dr. K. Jayasuriya", "specialty": "Pediatrician", "hospital": "Teaching Hospital", "city": "Kandy", "contact": "081-2222222"},
    {"name": "Dr. Nimali Silva", "specialty": "Neurologist", "hospital": "Karapitiya Teaching Hospital", "city": "Galle", "contact": "091-2232250"},
    {"name": "Dr. Roshan Dias", "specialty": "General Physician", "hospital": "Nawaloka Hospital", "city": "Colombo", "contact": "011-5577111"},
    {"name": "Dr. Fathima Riaz", "specialty": "ENT Surgeon", "hospital": "Lanka Hospitals", "city": "Colombo", "contact": "011-5430000"}
]

def find_doctor(specialty, city):
    # 1. Normalize Inputs
    search_term = specialty.lower()
    
    # 2. Smart Synonyms (Map Symptoms to Doctors)
    if "skin" in search_term or "rash" in search_term or "acne" in search_term: 
        search_term = "dermatologist"
    elif "heart" in search_term or "chest" in search_term: 
        search_term = "cardiologist"
    elif "child" in search_term or "baby" in search_term: 
        search_term = "pediatrician"
    
    city_term = city.lower().strip() if city else ""

    results = []
    for doc in SRI_LANKAN_DOCTORS:
        doc_specialty = doc["specialty"].lower()
        doc_city = doc["city"].lower()
        
        # Check if the doctor's specialty matches the inferred need
        if search_term in doc_specialty:
            # If a city was requested, match it. If not, show ALL doctors of that type.
            if city_term == "" or city_term in doc_city:
                results.append(doc)
    
    if not results:
        # Return empty string if no match found (so we don't spam the user)
        return ""
    
    # 3. Build Output Table
    output = f"\n\n---\n### 🏥 Recommended {specialty.capitalize()}s in Sri Lanka\n"
    for doc in results:
        output += f"- **{doc['name']}**\n"
        output += f"  - 🏥 {doc['hospital']} ({doc['city']})\n"
        output += f"  - 📞 `{doc['contact']}`\n"
    return output

# ==========================================
# 2. MEDICAL ADVICE RAG (Unstructured Data)
# ==========================================

@st.cache_resource
def setup_rag():
    print("Loading Medical Guidelines...")
    try:
        # Ensure you have 'data/medical_guidelines.pdf' in your project folder
        loader = PDFPlumberLoader("data/medical_guidelines.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
        return vectorstore
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

def get_medical_advice(query):
    vectorstore = setup_rag()
    if not vectorstore:
        return "Medical guidelines PDF not found. Using general knowledge."
        
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])