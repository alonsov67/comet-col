"""
APLICACIÓN PRINCIPAL (FRONTEND)
Responsabilidad: Orquestar la UI de Streamlit y llamar a los módulos.
No contiene lógica de negocio, solo presentación.
"""
import streamlit as st
import json
from modules.repository import TuvaRepository
from modules.tokenization import TokenizadorCoMET
from modules.engine import CometEngine

# --- Configuración de la Página ---
st.set_page_config(page_title="CoMET-Col Modular", layout="wide", page_icon="🧬")

# --- Inyección de Dependencias (Carga de Módulos) ---
@st.cache_resource
def cargar_sistema():
    repo = TuvaRepository()
    tokenizador = TokenizadorCoMET()
    engine = CometEngine()
    return repo, tokenizador, engine

try:
    repo, tokenizador, engine = cargar_sistema()
except Exception as e:
    st.error(f"Error crítico cargando módulos: {e}")
    st.stop()

# --- Interfaz de Usuario ---
st.title("🧬 CoMET-Col: Arquitectura Modular")
st.markdown("**Sistema de Auditoría Predictiva basado en Agentes.**")

# 1. Carga de Datos (Usando Repository Module)
with st.sidebar:
    st.header("🔧 Configuración")
    st.success("Módulos cargados correctamente")
    
    # Cargar datos desde repositorio
    hist_data, new_data, path_h, path_n = repo.cargar_datos()
    
    if not hist_data or not new_data:
        st.error("Faltan datos en /datos_rip")
    else:
        st.info(f"Histórico: {len(hist_data)} pacientes")
        st.caption(f"Fuente: {path_h}")

    modo_ver = st.toggle("Ver Tokens Semánticos", value=True)

# 2. Layout Principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Paciente Entrante (RIPS)")
    st.json(new_data, expanded=False)
    
    if st.button("🚀 Ejecutar Análisis", type="primary"):
        st.session_state['run_analysis'] = True

# 3. Ejecución del Flujo (Pipeline)
if st.session_state.get('run_analysis'):
    with st.spinner("Tokenizando y Vectorizando..."):
        # A. Tokenización (Usando Tokenization Module)
        secuencia_nuevo = tokenizador.construir_secuencia(new_data)
        
        # B. Vectorización Histórica (Usando Engine Module)
        # (Nota: En prod, esto estaría pre-calculado en ChromaDB, no en vivo)
        vectores_hist = []
        meta_hist = []
        for pt in hist_data:
            sec = tokenizador.construir_secuencia(pt)
            vec = engine.generar_embedding(sec)
            vectores_hist.append(vec)
            meta_hist.append({"id": pt['id'], "secuencia": sec})
            
        # C. Embedding Nuevo y Búsqueda
        vector_nuevo = engine.generar_embedding(secuencia_nuevo)
        idx, score = engine.buscar_similitud(vector_nuevo, vectores_hist)
        match_paciente = meta_hist[idx]

    # 4. Visualización de Resultados
    with col2:
        st.subheader("🧠 Visión CoMET")
        if modo_ver:
            # Formateo visual simple
            fmt = secuencia_nuevo.replace("DX:", "**DX:** ").replace("TIEMPO:", " ⏱️**TIEMPO:** ")
            st.info(fmt)
        
        st.subheader("🔍 Inferencia Vectorial")
        c1, c2 = st.columns(2)
        c1.metric("Similitud", f"{score:.1%}")
        c1.caption(f"Match Histórico: {match_paciente['id']}")
        
        if score > 0.8:
            c2.error("⚠️ Patrón de Alto Riesgo")
        else:
            c2.success("Patrón Estable")

    # 5. Predicción Agéntica (Usando Engine Module)
    st.markdown("---")
    st.subheader("🔮 Predicción del Agente")
    
    with st.spinner("Consultando Llama 3.1..."):
        prediccion = engine.predecir_riesgo(secuencia_nuevo, match_paciente['secuencia'])
        
        k1, k2, k3 = st.columns(3)
        riesgo = prediccion.get('riesgo', 'UNKNOWN')
        
        if 'ALTO' in str(riesgo).upper():
            k1.error(f"RIESGO: {riesgo}")
        else:
            k1.info(f"RIESGO: {riesgo}")
            
        k2.warning(f"Evento: {prediccion.get('evento_futuro')}")
        k3.metric("Tendencia", prediccion.get('costo_tendencia'))
        
        st.markdown(f"**Análisis:** {prediccion.get('explicacion')}")