# Sistema RAG Multi-Agente para CVs

Este es un sistema de consulta de CVs basado en RAG (Retrieval-Augmented Generation) que utiliza múltiples agentes para proporcionar respuestas personalizadas sobre diferentes CVs.

## Requisitos previos

- Python 3.8 o superior
- Cuenta en Pinecone (para el almacenamiento de vectores)
- Cuenta en Groq (para el modelo de lenguaje)

## Configuración

1. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar las variables de entorno:
- Copia el archivo `.env.example` a `.env`
- Rellena las claves API necesarias en el archivo `.env`:
  - `PINECONE_API_KEY`: Tu clave API de Pinecone
  - `GROQ_API_KEY`: Tu clave API de Groq

## Uso

1. Ejecutar la aplicación:
```bash
streamlit run cv_system.py
```

2. Acceder a la aplicación web a través del navegador (por defecto en http://localhost:8501)

3. Puedes:
- Subir nuevos CVs en formato texto
- Hacer preguntas sobre los CVs
- El sistema seleccionará automáticamente el agente apropiado según el contexto de la pregunta

## Características

- Procesamiento de múltiples CVs
- Búsqueda semántica utilizando embeddings
- Respuestas personalizadas según el perfil del CV
- Interfaz web intuitiva con Streamlit
