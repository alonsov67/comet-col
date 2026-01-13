"""
MÓDULO: KNOWLEDGE
Responsabilidad: Contener la ontología estática y definiciones del sistema de salud (SISPRO).
No tiene lógica compleja, solo diccionarios de traducción.
"""

class MaestroSispro:
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
        
        # REGIMEN
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