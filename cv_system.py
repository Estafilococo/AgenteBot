import streamlit as st
import os
import re
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Agent:
    def __init__(self, name: str, cv_id: str, personality: str):
        self.name = name
        self.cv_id = cv_id
        self.personality = personality
        
    def get_system_prompt(self) -> str:
        return f"""Eres un asistente especializado en analizar el CV de {self.name}. 
        {self.personality}
        Responde de manera precisa basándote en la información del CV."""

class ConditionalEdge:
    def __init__(self):
        self.name_patterns = {}
        
    def add_pattern(self, agent_name: str, patterns: list):
        """Añade patrones de reconocimiento para un agente."""
        self.name_patterns[agent_name] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def select_agent(self, query: str, default_agent: Agent, available_agents: Dict[str, Agent]) -> Agent:
        """Selecciona el agente apropiado basado en la query."""
        for agent_name, patterns in self.name_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return available_agents.get(agent_name, default_agent)
        return default_agent

class LocalStorage:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.vectors_file = self.base_path / "vectors.json"
        self.load_vectors()

    def load_vectors(self):
        if self.vectors_file.exists():
            with open(self.vectors_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {"vectors": [], "metadata": []}
            self.save_vectors()

    def save_vectors(self):
        with open(self.vectors_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def upsert(self, vectors, metadata):
        self.data["vectors"].extend([v.tolist() for v in vectors])
        self.data["metadata"].extend(metadata)
        self.save_vectors()

    def query(self, vector, top_k=3, filter_dict=None):
        if not self.data["vectors"]:
            return []

        # Convertir los vectores almacenados a numpy array
        stored_vectors = np.array(self.data["vectors"])
        query_vector = np.array(vector).reshape(1, -1)
        
        # Calcular similitud
        similarities = cosine_similarity(query_vector, stored_vectors)[0]
        
        # Obtener los índices de los top_k resultados
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filtrar por cv_id si es necesario
        results = []
        for idx in top_indices:
            metadata = self.data["metadata"][idx]
            if filter_dict is None or all(metadata.get(k) == v for k, v in filter_dict.items()):
                results.append({
                    "score": float(similarities[idx]),
                    "metadata": metadata
                })
        
        return results

class RAGSystem:
    def __init__(self, groq_api_key: str):
        self.storage = LocalStorage("data")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.groq_client = Groq(api_key=groq_api_key)
        self.agents: Dict[str, Agent] = {}
        self.default_agent: Optional[Agent] = None
        self.edge = ConditionalEdge()

    def extract_name_from_cv(self, content: str) -> str:
        """Extrae el nombre del contenido del CV."""
        first_lines = content.split('\n')[:5]
        for line in first_lines:
            if any(keyword in line.lower() for keyword in ['nombre:', 'name:', 'cv de']):
                name = line.lower().replace('nombre:', '').replace('name:', '').replace('cv de', '').strip()
                return ' '.join(word.capitalize() for word in name.split())
        
        from datetime import datetime
        return f"CV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def process_and_store_cv(self, content: str) -> str:
        """Procesa y almacena un nuevo CV."""
        nombre = self.extract_name_from_cv(content)
        cv_id = f"{nombre.lower().replace(' ', '_')}_cv"
        
        # Dividir el contenido en chunks
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        
        # Procesar cada chunk
        vectors = []
        metadata = []
        for i, chunk in enumerate(chunks):
            # Generar embedding
            embedding = self.embedding_model.encode(chunk)
            vectors.append(embedding)
            metadata.append({
                "cv_id": cv_id,
                "texto": chunk
            })
        
        # Almacenar vectores y metadata
        self.storage.upsert(vectors, metadata)
        
        # Crear y añadir nuevo agente
        new_agent = Agent(
            name=nombre,
            cv_id=cv_id,
            personality="Soy profesional y preciso en mis respuestas."
        )
        self.add_agent(
            new_agent,
            patterns=[nombre.lower()],
            is_default=(len(self.agents) == 0)
        )
        
        return cv_id

    def add_agent(self, agent: Agent, patterns: list, is_default: bool = False):
        """Añade un nuevo agente al sistema."""
        self.agents[agent.name] = agent
        self.edge.add_pattern(agent.name, patterns)
        if is_default:
            self.default_agent = agent

    def search_similar_info(self, query: str, cv_id: str, top_k: int = 3):
        """Busca información similar en el CV específico."""
        query_embedding = self.embedding_model.encode(query)
        results = self.storage.query(
            vector=query_embedding,
            top_k=top_k,
            filter_dict={"cv_id": cv_id}
        )
        return results

    def generate_response(self, query: str) -> str:
        """Genera una respuesta usando el agente apropiado."""
        selected_agent = self.edge.select_agent(query, self.default_agent, self.agents)
        
        if not selected_agent:
            return "No hay ningún CV cargado en el sistema. Por favor, sube un CV primero."
        
        results = self.search_similar_info(query, selected_agent.cv_id)
        
        if not results:
            return "No encontré información relevante para responder esta pregunta."
        
        context = results[0]["metadata"]["texto"]
        
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": selected_agent.get_system_prompt()},
                {"role": "user", "content": f"""
                Basándote en el siguiente contexto del CV:
                {context}
                
                Responde a la siguiente pregunta:
                {query}
                """}
            ]
        )
        
        return response.choices[0].message.content

def main():
    st.title("Sistema RAG Multi-Agente para CVs")
    
    # Configuración inicial
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("Falta configurar la clave API de Groq en el archivo .env")
        return
    
    # Inicializar sistema RAG
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem(groq_api_key)
    
    # Interface de usuario
    st.write("Puedes preguntar sobre los CVs de los alumnos. Si no especificas un alumno, se usará el CV por defecto.")
    
    # Subir nuevo CV
    with st.expander("Subir nuevo CV"):
        st.write("El nombre se extraerá automáticamente del contenido del CV.")
        cv_file = st.file_uploader("Sube el CV en formato texto", type=["txt"])
        if cv_file:
            content = cv_file.read().decode("utf-8")
            try:
                cv_id = st.session_state.rag_system.process_and_store_cv(content)
                st.success(f"CV procesado correctamente (ID: {cv_id})")
            except Exception as e:
                st.error(f"Error al procesar el CV: {str(e)}")
    
    # Área de preguntas
    query = st.text_input("¿Qué quieres saber sobre los CVs?")
    if query:
        try:
            response = st.session_state.rag_system.generate_response(query)
            st.write("### Respuesta:")
            st.write(response)
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {str(e)}")

if __name__ == "__main__":
    main()
