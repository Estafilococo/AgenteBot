import streamlit as st
import os
import re
from dotenv import load_dotenv
from groq import Groq
from typing import Dict, Optional

class Agent:
    def __init__(self, name: str, cv_id: str, personality: str):
        self.name = name
        self.cv_id = cv_id
        self.personality = personality
        
    def get_system_prompt(self) -> str:
        return f"""Eres un asistente especializado en analizar el CV de {self.name}. 
        {self.personality}
        Responde de manera precisa basándote en la información proporcionada."""

class ConditionalEdge:
    def __init__(self):
        self.patterns = {
            'experiencia': [
                r'(?i)experiencia.*(?:laboral|trabajo|profesional)',
                r'(?i)(?:donde|que empresas).*(?:trabajado|trabajó)',
                r'(?i)trayectoria.*profesional'
            ],
            'educacion': [
                r'(?i)(?:estudios|educación|formación)',
                r'(?i)(?:donde|que|cual).*(?:estudió|estudio|titulo)',
                r'(?i)(?:universidad|carrera|grado)'
            ],
            'habilidades': [
                r'(?i)(?:habilidades|skills|competencias)',
                r'(?i)que.*(?:sabe hacer|conoce|domina)',
                r'(?i)(?:tecnologías|herramientas)'
            ],
            'personal': [
                r'(?i)(?:nombre|edad|ubicación)',
                r'(?i)(?:quien|quién) es',
                r'(?i)datos.*personales'
            ]
        }
        
    def select_context(self, query: str) -> str:
        """Selecciona el contexto apropiado basado en la query."""
        for context, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return context
        return 'general'  # Contexto por defecto

# Inicializar Groq
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)
edge = ConditionalEdge()

def process_query(cv_content: str, query: str) -> str:
    """Procesa una pregunta sobre el CV usando Groq."""
    try:
        # Determinar el contexto de la pregunta
        context = edge.select_context(query)
        
        # Ajustar el prompt según el contexto
        context_prompts = {
            'experiencia': "Enfócate en la experiencia laboral y trayectoria profesional.",
            'educacion': "Enfócate en la formación académica y estudios.",
            'habilidades': "Enfócate en las habilidades técnicas y competencias.",
            'personal': "Enfócate en la información personal y datos de contacto.",
            'general': "Proporciona una respuesta general basada en todo el CV."
        }
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": f"""Eres un asistente especializado en analizar CVs. 
                {context_prompts[context]}
                Responde de manera precisa basándote en la información proporcionada."""},
                {"role": "user", "content": f"""
                Basándote en el siguiente CV:
                {cv_content}
                
                Responde a esta pregunta:
                {query}
                """}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al procesar la pregunta: {str(e)}"

def main():
    st.title("Sistema de Consulta de CVs")
    
    # Variables de estado para el CV
    if 'cv_content' not in st.session_state:
        st.session_state.cv_content = None
    
    # Subir CV
    with st.expander("Subir nuevo CV"):
        cv_file = st.file_uploader("Sube el CV en formato texto", type=["txt"])
        if cv_file:
            st.session_state.cv_content = cv_file.read().decode("utf-8")
            st.success("CV cargado correctamente")
    
    # Área de preguntas
    if st.session_state.cv_content:
        st.write("---")
        st.write("Puedes hacer preguntas sobre el CV cargado.")
        query = st.text_input("¿Qué quieres saber sobre el CV?")
        
        if query:
            with st.spinner("Analizando el CV..."):
                response = process_query(st.session_state.cv_content, query)
                st.write("### Respuesta:")
                st.write(response)
                
                # Mostrar el contexto detectado (para debugging)
                context = edge.select_context(query)
                st.write(f"*Contexto detectado: {context}*")
    else:
        st.info("Por favor, sube un CV primero para poder hacer preguntas sobre él.")

if __name__ == "__main__":
    main()
