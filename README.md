🧬 CoMET-Col: Arquitectura Agéntica para Auditoría en SaludDetección de Fugas Financieras por Fragmentación Clínica🚧 1. El Dolor: La "Caja Negra" de la FragmentaciónEn el modelo actual de pago por evento, las complicaciones derivadas de una atención deficiente se facturan como eventos nuevos e independientes.El Problema: Una apendicectomía en la IPS A y una infección post-quirúrgica en la IPS B no se "hablan" administrativamente.El Impacto: El sistema paga doble: Paga por la cirugía inicial y paga por la complicación que debió prevenirse.💡 2. La Solución: CoMET-ColCoMET-Col (Colombia Medical Event Transformer) es un motor de auditoría predictiva basado en la metodología de Medical Tokenization (Epic Systems/Microsoft, 2025), adaptado a la ontología de FEV-RIPS y SISPRO.Diferencial TecnológicoA diferencia de los validadores de reglas estáticas (IF diagnosis == X), CoMET-Col utiliza Embeddings Semánticos para entender trayectorias clínicas.Tokenización Semántica: Convierte JSON de RIPS en narrativas clínicas.Vectorización: Entiende que \[K358] seguido de \[GAP\_5\_DIAS] y \[T814] implica una causalidad clínica (fragmentación), no una coincidencia.Agentes de IA: Utiliza LLMs (Llama 3.1) para razonar sobre la evidencia y estimar costos futuros.🏗️ 3. Arquitectura ModularEl proyecto sigue una arquitectura limpia para separar la ontología médica de la lógica de inteligencia artificial.graph LR

&nbsp;   A\[JSON RIPS] --> B(Tokenization Module)

&nbsp;   B --> C{Engine Module}

&nbsp;   C -->|Vectores| D\[ChromaDB]

&nbsp;   C -->|Inferencia| E\[Agente Llama 3.1]

&nbsp;   E --> F\[UI Streamlit]

&nbsp;   G\[Knowledge Module] -.->|Ontología SISPRO| B

Estructura del Proyectomodules/knowledge.py: Ontología estática (CIE-10, CUPS, Medicamentos).modules/tokenization.py: Algoritmo de transformación de eventos discretos a secuencias.modules/engine.py: Motor de IA (Ollama + Embeddings).modules/repository.py: Capa de persistencia (Simulación Data Warehouse).app.py: Orquestador de Interfaz Gráfica.🚀 4. Instalación y UsoEste proyecto está diseñado para ejecutarse localmente garantizando la privacidad de los datos (Habeas Data).PrerrequisitosAnaconda (Python 3.10+)Ollama instalado y ejecutándose.Paso a pasoClonar el repositorio:git clone \[https://github.com/alonsov67/comet-col.git](https://github.com/alonsov67/comet-col.git)

cd comet-col

Preparar el entorno:conda create -n salud\_ai python=3.10

conda activate salud\_ai

pip install -r requirements.txt

Descargar modelos de IA (Local):ollama pull llama3.1

ollama pull nomic-embed-text

Ejecutar la Plataforma:Asegúrate de tener ollama serve corriendo en otra terminal.streamlit run app.py

🗺️ 5. Hoja de Ruta (Roadmap)FaseEstadoDescripciónFase 1: Mockup Funcional✅ CompletadoEjecución local con Llama 3.1, LangChain y RAG básico sobre JSON simulados.Fase 2: Embeddings de Dominio🚧 En ProcesoEntrenamiento de modelo específico con ontología colombiana completa (CIE-10 + CUPS + Manual Tarifario).Fase 3: Despliegue Cloud☁️ FuturoMigración a infraestructura segura (Azure/AWS) y conexión con APIs reales de interoperabilidad FHIR.Desarrollado para el ecosistema de innovación en salud de Colombia.

