"""
GUÍA DE EJECUCIÓN MODO LOCAL (WINDOWS / ANACONDA):

1. Abre Anaconda Prompt.
2. Navega a la carpeta 'plataforma_auditoria'.
3. Activa el entorno: 
   conda activate salud_ai
4. Asegúrate de tener el servidor Ollama en ejecución con el modelo Llama 3.1.
5. Ejecuta:
   streamlit run comet_col_embedding.py

NOTA: Al ejecutar, el script buscará la carpeta 'datos_rip' con los JSON de ejemplo.
"""

import json
import os
import numpy as np
import streamlit as st
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import JsonOutputParser

# ==========================================
# 0. GESTIÓN DE DATOS EXTERNOS (TUVA MOCK)
# ==========================================
def inicializar_tuva_mock():
    """
    Simula la conexión a un Data Warehouse (Tuva) mediante archivos JSON locales.
    Si no existen, crea archivos base para que el usuario pueda editarlos.
    """
    # Definir ruta relativa a la carpeta del script
    folder_path = "datos_rip"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    path_historico = os.path.join(folder_path, "historial_paciente.json")
    path_nuevo = os.path.join(folder_path, "nuevo_evento.json")
    
    # 1. Crear Histórico Base si no existe
    if not os.path.exists(path_historico):
        data_historica = [
            {
                "id": "PT_INTEGRAL_01",
                "perfil": {"sexo": "M", "edad": 55, "regimen": "Contributivo", "tipo_afiliado": "Cotizante"},
                "eventos": [
                    {"fecha": "2024-01-10", "cod_ips": "IPS_A", "especialidad_medico": "MED_GENERAL", "diagnosticos": [{"cod": "E119", "desc": "Diabetes 2"}], "medicamentos": [{"atc": "A10BA02", "desc": "Metformina"}]},
                    {"fecha": "2024-04-12", "cod_ips": "IPS_A", "especialidad_medico": "MED_INTERNA", "diagnosticos": [{"cod": "E119", "desc": "Diabetes 2"}], "procedimientos": [{"cod": "903895", "desc": "Creatinina"}]}
                ]
            },
            {
                "id": "PT_RIESGO_RENAL_02",
                "perfil": {"sexo": "M", "edad": 58, "regimen": "Contributivo", "tipo_afiliado": "Cotizante"},
                "eventos": [
                    {"fecha": "2024-01-15", "cod_ips": "IPS_A", "especialidad_medico": "URGENCIAS", "diagnosticos": [{"cod": "E119", "desc": "Diabetes 2"}], "medicamentos": []},
                    {"fecha": "2024-06-20", "cod_ips": "IPS_B", "especialidad_medico": "NEFROLOGIA", "diagnosticos": [{"cod": "N183", "desc": "Enfermedad Renal"}], "procedimientos": [{"cod": "903895", "desc": "Creatinina"}]}
                ]
            }
        ]
        with open(path_historico, 'w', encoding='utf-8') as f:
            json.dump(data_historica, f, indent=2, ensure_ascii=False)
            
    # 2. Crear Caso Nuevo Base si no existe
    if not os.path.exists(path_nuevo):
        data_nuevo = {
            "id": "PT_NUEVO_ALTO_COSTO",
            "perfil": {"sexo": "M", "edad": 60, "regimen": "Contributivo", "tipo_afiliado": "Beneficiario"},
            "eventos": [
                {"fecha": "2025-01-05", "cod_ips": "IPS_C", "especialidad_medico": "URGENCIAS", "diagnosticos": [{"cod": "E10", "desc": "Diabetes"}], "medicamentos": []},
                {"fecha": "2025-05-10", "cod_ips": "IPS_D", "especialidad_medico": "URGENCIAS", "diagnosticos": [{"cod": "E105", "desc": "Diabetes complicada"}]}
            ]
        }
        with open(path_nuevo, 'w', encoding='utf-8') as f:
            json.dump(data_nuevo, f, indent=2, ensure_ascii=False)

    return path_historico, path_nuevo

# ==========================================
# 1. ONTOLOGÍA COLOMBIANA (MAESTROS SISPRO)
# ==========================================
class MaestroSispro:
    """
    Simula la base de conocimientos del Ministerio de Salud (SISPRO).
    Traduce códigos crudos a conceptos semánticos ricos para el embedding.
    """
    def __init__(self):
        # CIE-10: Diagnósticos
        self.cie10 = {
            "E10": "DIABETES MELLITUS INSULINODEPENDIENTE TIPO 1 ENDOCRINO",
            "E119": "DIABETES MELLITUS TIPO 2 NO INSULINODEPENDIENTE SIN COMPLICACIONES METABOLICO",
            "E105": "DIABETES MELLITUS TIPO 1 CON COMPLICACIONES CIRCULATORIAS PERIFERICAS",
            "N183": "ENFERMEDAD RENAL CRONICA ETAPA 3 FALLA RENAL MODERADA FILTRACION GLOMERULAR DISMINUIDA",
            "I10X": "HIPERTENSION ARTERIAL ESENCIAL PRIMARIA RIESGO CARDIOVASCULAR",
            "T814": "INFECCION CONSECUTIVA A PROCEDIMIENTO HERIDA QUIRURGICA COMPLICACION POSOPERATORIA",
            "Z000": "EXAMEN MEDICO GENERAL CONTROL PREVENTIVO SALUD"
        }
        
        # CUPS: Procedimientos
        self.cups = {
            "903895": "CREATININA EN SUERO ORINA FUNCION RENAL QUIMICA SANGUINEA",
            "903841": "HEMOGLOBINA GLICOSILADA HB1AC CONTROL DIABETES",
            "890201": "CONSULTA DE PRIMERA VEZ POR MEDICINA GENERAL",
            "890301": "CONSULTA DE CONTROL POR MEDICINA GENERAL",
            "871010": "RADIOGRAFIA DE TORAX",
            "881112": "ECOGRAFIA RENAL VIAL URINARIAS"
        }
        
        # ATC: Medicamentos
        self.atc = {
            "A10BA02": "METFORMINA ANTIDIABETICO ORAL BIGUANIDAS",
            "A10A": "INSULINAS Y ANALOGOS HORMONA",
            "C09AA02": "ENALAPRIL ANTIHIPERTENSIVO INHIBIDOR ECA",
            "J01CR02": "AMOXICILINA Y INHIBIDOR DE ENZIMA ANTIBIOTICO PENICILINAS"
        }
        
        # REGIMEN: Contexto Financiero
        self.regimen = {
            "CONTRIBUTIVO": "PAGO POR CAPACIDAD ASEGURAMIENTO PRIVADO LABORAL",
            "SUBSIDIADO": "PAGO POR ESTADO SISBEN VULNERABILIDAD",
            "ESPECIAL": "FUERZAS MILITARES MAGISTERIO ECOPETROL"
        }

    def get_concepto(self, tipo, codigo):
        codigo_limpio = codigo.replace(".", "")
        if tipo == "DX":
            return self.cie10.get(codigo_limpio, "ENFERMEDAD_NO_ESPECIFICADA")
        elif tipo == "PROC":
            return self.cups.get(codigo_limpio, "PROCEDIMIENTO_NO_ESPECIFICADO")
        elif tipo == "MED":
            return self.atc.get(codigo_limpio, "MEDICAMENTO_NO_ESPECIFICADO")
        elif tipo == "REG":
            return self.regimen.get(codigo.upper(), "REGIMEN_NO_ESPECIFICADO")
        return codigo

# ==========================================
# 2. TOKENIZADOR MÉDICO (LA LÓGICA CoMET)
# ==========================================
class TokenizadorCoMET:
    def __init__(self):
        self.maestro = MaestroSispro()

    def calcular_gap_temporal(self, fecha_prev, fecha_curr):
        if not fecha_prev:
            return "[INICIO_HISTORIA]"
        
        d1 = datetime.strptime(fecha_prev, "%Y-%m-%d")
        d2 = datetime.strptime(fecha_curr, "%Y-%m-%d")
        dias = (d2 - d1).days
        
        if dias == 0: return "[MISMO_DIA_URGENCIA]"
        if dias <= 7: return "[SEMANA_1_SEGUIMIENTO]"
        if dias <= 30: return "[MES_1_CONTROL]"
        if dias <= 90: return "[TRIMESTRE_1_CRONICO]"
        return f"[GAP_LARGO_{dias}_DIAS_ABANDONO]"

    def construir_secuencia(self, paciente_data):
        perfil = paciente_data['perfil']
        eventos = sorted(paciente_data['eventos'], key=lambda x: x['fecha'])
        
        semantica_regimen = self.maestro.get_concepto("REG", perfil['regimen'])
        secuencia = [
            f"PACIENTE_SEXO:{perfil['sexo']}",
            f"EDAD:{perfil['edad']}_ANOS_GRUPO_RIESGO",
            f"CONTEXTO_FINANCIERO:{semantica_regimen}"
        ]
        
        fecha_anterior = None
        
        for evt in eventos:
            time_token = self.calcular_gap_temporal(fecha_anterior, evt['fecha'])
            secuencia.append(f"TIEMPO:{time_token}")
            secuencia.append(f"LUGAR_ATENCION:IPS_{evt['cod_ips']}") 
            secuencia.append(f"ACTOR_MEDICO:{evt['especialidad_medico']}")
            
            if 'diagnosticos' in evt:
                for dx in evt['diagnosticos']:
                    desc_rica = self.maestro.get_concepto("DX", dx['cod'])
                    secuencia.append(f"DX:{dx['cod']}__{desc_rica.replace(' ', '_')}")
            
            if 'procedimientos' in evt:
                for proc in evt['procedimientos']:
                    desc_rica = self.maestro.get_concepto("PROC", proc['cod'])
                    secuencia.append(f"PROC:{proc['cod']}__{desc_rica.replace(' ', '_')}")
            
            if 'medicamentos' in evt:
                for med in evt['medicamentos']:
                    desc_rica = self.maestro.get_concepto("MED", med['atc'])
                    secuencia.append(f"FARMACO:{med['atc']}__{desc_rica.replace(' ', '_')}")
            
            fecha_anterior = evt['fecha']
        
        return " ".join(secuencia)

# ==========================================
# 4. INTERFAZ GRÁFICA (STREAMLIT APP)
# ==========================================

def main():
    st.set_page_config(page_title="CoMET-Col Auditoría Predictiva", layout="wide", page_icon="🧬")
    
    # --- Inicialización de Datos Externos ---
    path_hist, path_new = inicializar_tuva_mock()
    
    st.title("🧬 CoMET-Col: Radar Predictivo de Salud")
    st.markdown("""
      Motor de Embeddings Semánticos para FEV-RIPS.  
    Transforma códigos administrativos en trayectorias clínicas para identificar riesgo por fragmentación / alto costo.
    """)
    
    # --- Cargar Modelos (Cacheado con validación JSON) ---
    @st.cache_resource
    def cargar_motor():
        return TokenizadorCoMET(), OllamaEmbeddings(model="nomic-embed-text"), ChatOllama(model="llama3.1", temperature=0.1, format="json")

    with st.spinner("Inicializando Motor Neuronal (Ollama)..."):
        try:
            tokenizador, embeddings_model, llm = cargar_motor()
        except Exception as e:
            st.error(f"Error conectando con Ollama: {e}. Asegúrate de tener 'ollama serve' corriendo.")
            return

    # --- Sidebar ---
    with st.sidebar:
        st.header("🔧 Panel de Control")
        st.info("Modelo de Embeddings: nomic-embed-text")
        st.info("Modelo de Razonamiento: Llama 3.1 (JSON Mode)")
        
        st.markdown("---")
        st.subheader("📂 Fuente de Datos (datos_rip)")
        
        # Carga de Histórico
        try:
            with open(path_hist, 'r', encoding='utf-8') as f:
                base_datos_pacientes = json.load(f)
            st.success(f"✅ Histórico cargado ({len(base_datos_pacientes)} pacientes)")
            st.caption(f"Ruta: {path_hist}")
        except FileNotFoundError:
            st.error(f"❌ No se encontró: {path_hist}")
            base_datos_pacientes = []

        # Carga de Caso Nuevo
        try:
            with open(path_new, 'r', encoding='utf-8') as f:
                caso_nuevo = json.load(f)
            st.success("✅ Caso nuevo cargado")
            st.caption(f"Ruta: {path_new}")
        except FileNotFoundError:
            st.error(f"❌ No se encontró: {path_new}")
            caso_nuevo = {}

        st.markdown("---")
        modo_ver = st.toggle("Modo Depuración (Ver Tokens)", value=True)

    # --- Layout Principal ---
    col1, col2 = st.columns([1, 1])

    if not caso_nuevo or not base_datos_pacientes:
        st.warning("Por favor verifica que los archivos JSON existan en la carpeta 'datos_rip'.")
        return

    with col1:
        st.subheader("📂 1. Input: RIPS Estructurado (JSON)")
        st.json(caso_nuevo, expanded=False)
        
        if st.button("🚀 Ejecutar Análisis CoMET", type="primary"):
            st.session_state['ejecutado'] = True

    if st.session_state.get('ejecutado'):
        # 1. Tokenización y Vectorización
        secuencia_nuevo = tokenizador.construir_secuencia(caso_nuevo)
        vector_nuevo = embeddings_model.embed_query(secuencia_nuevo)
        
        # Procesar base histórica
        vectores_historicos = []
        metadata_historica = []
        for pt in base_datos_pacientes:
            sec = tokenizador.construir_secuencia(pt)
            vec = embeddings_model.embed_query(sec)
            vectores_historicos.append(vec)
            metadata_historica.append({"id": pt['id'], "secuencia": sec})

        # 2. Búsqueda Vectorial
        similitudes = cosine_similarity([vector_nuevo], vectores_historicos)[0]
        idx_mas_similar = np.argmax(similitudes)
        paciente_similar = metadata_historica[idx_mas_similar]
        score_similitud = similitudes[idx_mas_similar]

        with col2:
            st.subheader("🧠 2. Visión CoMET (Tokens Semánticos)")
            if modo_ver:
                tokens_fmt = secuencia_nuevo \
                    .replace("DX:", "**DX:** ") \
                    .replace("TIEMPO:", " ⏱️**TIEMPO:** ") \
                    .replace("LUGAR_ATENCION:", " 🏥**LUGAR:** ")
                st.info(tokens_fmt)
            else:
                st.text("Secuencia vectorizada oculta.")

            st.markdown("---")
            st.subheader("🔍 3. Inferencia Vectorial")
            c1, c2 = st.columns(2)
            c1.metric("Similitud con Historia", f"{score_similitud:.1%}")
            c1.caption(f"Match: {paciente_similar['id']}")
            
            if score_similitud > 0.8:
                c2.error("⚠️ Patrón de Alto Riesgo")
            else:
                c2.success("Patrón Estable")

        # 3. Predicción Agéntica
        st.markdown("---")
        st.subheader("🔮 4. Predicción de Futuro y Costos (Agente Llama 3.1)")
        
        with st.spinner("Generando escenario futuro..."):
            prompt = f"""
            Eres CoMET-Col, experto en riesgo salud Colombia.
            
            PACIENTE ACTUAL (Tokens): {secuencia_nuevo}
            HISTORIA SIMILAR (Tokens): {paciente_similar['secuencia']}
            
            Predice riesgo de Alto Costo (Diálisis/UCI) en 6 meses.
            Responde ÚNICAMENTE en JSON válido con este formato:
            {{ "riesgo": "ALTO/MEDIO/BAJO", "evento_futuro": "string", "costo_tendencia": "string", "explicacion": "string" }}
            """
            
            # BLOQUE ROBUSTO DE PREDICCIÓN
            try:
                res = llm.invoke(prompt)
                
                # Intentamos parsear el JSON
                parser = JsonOutputParser()
                prediccion = parser.parse(res.content)
                
                # Visualización
                k1, k2, k3 = st.columns(3)
                riesgo = prediccion.get('riesgo', 'DESC')
                
                if 'ALTO' in str(riesgo).upper():
                    k1.error(f"RIESGO: {riesgo}")
                else:
                    k1.info(f"RIESGO: {riesgo}")
                    
                k2.warning(f"Evento: {prediccion.get('evento_futuro')}")
                k3.metric("Tendencia Costo", prediccion.get('costo_tendencia'))
                
                st.markdown(f"**Justificación Clínica:** {prediccion.get('explicacion')}")
                
            except Exception as e:
                st.error("Error interpretando la respuesta de la IA.")
                st.warning("Detalle técnico del error para depuración:")
                st.code(str(e))
                if 'res' in locals():
                    st.warning("Respuesta cruda recibida del modelo (puede no ser JSON válido):")
                    st.text(res.content)

if __name__ == "__main__":
    main()