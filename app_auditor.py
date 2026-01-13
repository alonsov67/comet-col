import streamlit as st
import json
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Auditoría Agéntica FEV-RIPS", layout="wide")

st.title("🏥 Plataforma de Análisis de Costos y Fragmentación")
st.markdown("### Auditoría Inteligente sobre FEV-RIPS (HL7 FHIR / JSON)")

# ---------------------------------------------------------
# CARGA DE MODELOS (Cacheada para no recargar cada vez)
# ---------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    print(">>> Iniciando modelos Ollama...")
    llm = ChatOllama(model="llama3.1", temperature=0, format="json")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return llm, embeddings

llm, embeddings = cargar_modelos()

# ---------------------------------------------------------
# BARRA LATERAL: CARGA DE DATOS
# ---------------------------------------------------------
st.sidebar.header("📁 Ingesta de Datos (JSON)")

# Simulación de lectura de archivos externos
try:
    with open('datos_rips/historial_paciente.json', 'r') as f:
        historial_data = json.load(f)
    with open('datos_rips/nuevo_evento.json', 'r') as f:
        nuevo_evento_data = json.load(f)
    st.sidebar.success("Archivos JSON cargados correctamente")
except FileNotFoundError:
    st.sidebar.error("No se encontraron los archivos en /datos_rips")
    st.stop()

# ---------------------------------------------------------
# VISUALIZACIÓN DE LOS DATOS (Tu requerimiento visual)
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.info("📂 Historial Clínico Administrativo (Tuva/DB)")
    df_historial = pd.DataFrame(historial_data)
    # Mostramos tabla limpia
    st.dataframe(df_historial[['fecha', 'prestador', 'descripcion', 'valor_neto']], hide_index=True)
    st.caption(f"Total registros históricos: {len(df_historial)}")

with col2:
    st.warning("📄 Nuevo FEV-RIPS a Auditar")
    # Visualización tipo "Tarjeta"
    st.json(nuevo_evento_data)
    st.metric(label="Valor Facturado", value=f"${nuevo_evento_data['valor_neto']:,.0f} COP")

st.divider()

# ---------------------------------------------------------
# LÓGICA AGÉNTICA (RAG + LLM)
# ---------------------------------------------------------

if st.button("🔍 Ejecutar Análisis de Fragmentación (Agente IA)", type="primary"):
    
    with st.spinner('Vectorizando historial y buscando relaciones semánticas...'):
        # 1. Preparar Documentos para Vector Store
        docs = []
        for evento in historial_data:
            content = f"{evento['descripcion']} (CIE10: {evento['cod_diagnostico']}) - IPS: {evento['prestador']} - Fecha: {evento['fecha']}"
            meta = {"id": evento['id_evento'], "valor": evento['valor_neto'], "ips": evento['prestador']}
            docs.append(Document(page_content=content, metadata=meta))
        
        # 2. Crear Vector Store temporal (Memoria)
        vector_db = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="auditoria_temp")
        
        # 3. Búsqueda Semántica (Retrieval)
        query = f"{nuevo_evento_data['cod_diagnostico']} {nuevo_evento_data['descripcion']}"
        resultados = vector_db.similarity_search(query, k=2)
        
        contexto_encontrado = "\n".join([f"- {doc.page_content}" for doc in resultados])
    
    with st.spinner('El Agente Auditor está analizando el caso...'):
        # 4. Definir Estructura de Salida
        class AuditoriaResult(BaseModel):
            es_fragmentacion: bool = Field(description="True si es atención fragmentada")
            causa_raiz: str = Field(description="El evento histórico que originó este nuevo cobro")
            explicacion: str = Field(description="Razonamiento clínico-administrativo")
            ahorro_potencial: float = Field(description="Estimación de porcentaje de ahorro si hubiera sido integral")

        parser = JsonOutputParser(pydantic_object=AuditoriaResult)

        # 5. Prompt del Agente
        prompt = PromptTemplate(
            template="""
            Analiza la siguiente situación de facturación médica en Colombia.
            
            HISTORIAL RELACIONADO ENCONTRADO (Base de Datos):
            {contexto}
            
            NUEVA FACTURA (RIPS):
            Fecha: {fecha_new}
            IPS: {ips_new}
            Procedimiento/Dx: {desc_new}
            
            Instrucciones:
            1. Identifica si la nueva factura es una complicación derivada del historial.
            2. Verifica si la IPS es diferente (Fragmentación de red).
            3. Genera el reporte en JSON.
            
            {format_instructions}
            """,
            input_variables=["contexto", "fecha_new", "ips_new", "desc_new"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm | parser
        
        respuesta = chain.invoke({
            "contexto": contexto_encontrado,
            "fecha_new": nuevo_evento_data['fecha'],
            "ips_new": nuevo_evento_data['prestador'],
            "desc_new": nuevo_evento_data['descripcion']
        })

    # ---------------------------------------------------------
    # VISUALIZACIÓN DE RESULTADOS
    # ---------------------------------------------------------
    st.subheader("Resultado de la Auditoría")
    
    c1, c2, c3 = st.columns(3)
    
    if respuesta['es_fragmentacion']:
        c1.error("🚨 DETECCIÓN: ATENCIÓN FRAGMENTADA")
    else:
        c1.success("✅ ATENCIÓN CORRECTA")
        
    c2.info(f"Causa Raíz Detectada: {respuesta['causa_raiz']}")
    
    # Calcular costo total involucrado (Histórico recuperado + Nuevo)
    costo_previo = sum([doc.metadata['valor'] for doc in resultados])
    costo_total_episodio = costo_previo + nuevo_evento_data['valor_neto']
    
    c3.metric("Costo Total del Episodio (Real)", f"${costo_total_episodio:,.0f}")

    st.markdown(f"**Razonamiento del Agente:**")
    st.write(respuesta['explicacion'])

    # Expander para ver la evidencia técnica
    with st.expander("Ver Evidencia Técnica (Embeddings Recuperados)"):
        st.write("El sistema encontró estos eventos previos como causantes:")
        for doc in resultados:
            st.code(doc.page_content)