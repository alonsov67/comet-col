"""
MÓDULO: ENGINE
Responsabilidad: Interacción con Modelos de Lenguaje (LLM) y Vectores.
Aísla la dependencia de Ollama/LangChain.
"""
import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import JsonOutputParser

class CometEngine:
    def __init__(self):
        # Inicialización de modelos
        self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="llama3.1", temperature=0.1, format="json")
        self.parser = JsonOutputParser()

    def generar_embedding(self, texto):
        return self.embeddings_model.embed_query(texto)

    def buscar_similitud(self, vector_query, lista_vectores):
        """Retorna índice y score del más similar."""
        if not lista_vectores:
            return -1, 0.0
        similitudes = cosine_similarity([vector_query], lista_vectores)[0]
        idx_max = np.argmax(similitudes)
        return idx_max, similitudes[idx_max]

    def predecir_riesgo(self, secuencia_actual, secuencia_similar):
        prompt = f"""
        Eres CoMET-Col, experto en riesgo salud Colombia.
        
        PACIENTE ACTUAL (Tokens): {secuencia_actual}
        HISTORIA SIMILAR (Tokens): {secuencia_similar}
        
        Predice riesgo de Alto Costo (Diálisis/UCI) en 6 meses.
        Responde ÚNICAMENTE en JSON válido con este formato:
        {{ "riesgo": "ALTO/MEDIO/BAJO", "evento_futuro": "string", "costo_tendencia": "string", "explicacion": "string" }}
        """
        try:
            res = self.llm.invoke(prompt)
            return self.parser.parse(res.content)
        except Exception as e:
            return {"riesgo": "ERROR", "explicacion": str(e)}