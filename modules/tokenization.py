"""
MÓDULO: TOKENIZATION
Responsabilidad: Implementar la lógica del paper CoMET.
Convierte eventos discretos y fechas en una secuencia narrativa semántica.
"""
from datetime import datetime
from modules.knowledge import MaestroSispro

class TokenizadorCoMET:
    def __init__(self):
        self.maestro = MaestroSispro()

    def _calcular_gap_temporal(self, fecha_prev, fecha_curr):
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
            time_token = self._calcular_gap_temporal(fecha_anterior, evt['fecha'])
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