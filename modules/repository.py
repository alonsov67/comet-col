"""
MÓDULO: REPOSITORY
Responsabilidad: Gestión de persistencia y acceso a datos (JSON/Archivos).
Simula la conexión con el Data Warehouse (Tuva).
"""
import json
import os

class TuvaRepository:
    def __init__(self, data_folder="datos_rip"):
        self.folder_path = data_folder
        self._inicializar_estructura()

    def _inicializar_estructura(self):
        """Crea la carpeta y archivos base si no existen."""
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def get_rutas(self):
        return (
            os.path.join(self.folder_path, "historial_paciente.json"),
            os.path.join(self.folder_path, "nuevo_evento.json")
        )

    def cargar_datos(self):
        path_hist, path_new = self.get_rutas()
        
        historico = []
        nuevo = {}
        
        # Carga segura con manejo de errores
        if os.path.exists(path_hist):
            with open(path_hist, 'r', encoding='utf-8') as f:
                historico = json.load(f)
        
        if os.path.exists(path_new):
            with open(path_new, 'r', encoding='utf-8') as f:
                nuevo = json.load(f)
                
        return historico, nuevo, path_hist, path_new