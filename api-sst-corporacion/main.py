"""
API SST — CORPORACIÓN E. WONG
FastAPI backend para predicción de accidentes laborales
Instalar: pip install fastapi uvicorn scikit-learn pandas numpy
Ejecutar: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pickle, json, numpy as np, uuid
from datetime import datetime

app = FastAPI(
    title="API SST — Corporación E. Wong",
    description="Sistema predictivo de accidentes laborales",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Cargar modelo
modelo = features = None
try:
    with open("outputs_modelo/modelo_rf.pkl","rb") as f: modelo = pickle.load(f)
    with open("outputs_modelo/features.json")    as f: features = json.load(f)
    print(f"Modelo cargado — {len(features)} variables")
except Exception as e:
    print(f"Modelo no encontrado, modo demo: {e}")
    features = []

UNIDAD_MAP = {"Gelan":0,"Mediterráneo":1,"Don Buffet":2,"Dansa":3,
              "Willax":4,"Mall del Sur":5,"Plaza Norte":6}
RIESGO_MAP = {"Bajo":0,"Medio":1,"Alto":2}
TURNO_MAP  = {"Mañana":0,"Tarde":1,"Noche":2,"Rotativo":3}
historial  = []

class DatosTrabajador(BaseModel):
    id_trabajador:         str
    unidad_negocio:        str
    area:                  str
    cargo:                 str
    antiguedad_meses:      int   = Field(ge=0, default=12)
    edad:                  int   = Field(ge=18, le=80, default=30)
    turno:                 str   = "Mañana"
    nivel_riesgo_puesto:   str   = "Medio"
    horas_trabajadas_mes:  int   = Field(ge=0, default=160)
    horas_extra_mes:       int   = Field(ge=0, default=0)
    n_cap_ult_12m:         int   = Field(ge=0, default=2)
    cap_reciente_30d:      int   = Field(ge=0, le=1, default=0)
    induccion_sst:         int   = Field(ge=0, le=1, default=1)
    cumple_epp:            int   = Field(ge=0, le=1, default=1)
    supervision_frecuente: int   = Field(ge=0, le=1, default=1)
    n_hall_criticos_mes:   int   = Field(ge=0, default=0)
    n_actos_sub_trab_mes:  int   = Field(ge=0, default=0)
    n_cond_sub_area_mes:   int   = Field(ge=0, default=0)
    n_inc_previos_12m:     int   = Field(ge=0, default=0)
    n_acc_previos_12m:     int   = Field(ge=0, default=0)
    dias_ult_incidente:    int   = Field(ge=0, default=999)
    dias_ult_accidente:    int   = Field(ge=0, default=999)
    supervisor_id:         Optional[str] = None
    observaciones:         Optional[str] = None

def prob_demo(d):
    return min(0.95, max(0.05,
        (0.15 if d.nivel_riesgo_puesto=="Alto" else 0.07 if d.nivel_riesgo_puesto=="Medio" else 0.03)+
        (0.10 if d.cumple_epp==0 else 0)+
        (0.10 if d.antiguedad_meses<6 else 0.05 if d.antiguedad_meses<12 else 0)+
        (0.08 if d.n_hall_criticos_mes>=2 else 0)+
        (0.08 if d.n_acc_previos_12m>0 else 0)+
        (0.05 if d.cap_reciente_30d==0 else 0)+
        (0.04 if d.horas_extra_mes>20 else 0)+
        (0.03 if d.turno=="Noche" else 0)))

def nivel_y_color(p):
    if p<0.20: return "BAJO",    "verde",   "Riesgo controlado. Mantener controles actuales."
    if p<0.40: return "MEDIO",   "amarillo","Riesgo moderado. Reforzar supervisión y EPP."
    if p<0.65: return "ALTO",    "naranja", "Riesgo alto. Intervención preventiva recomendada."
    return           "CRÍTICO", "rojo",    "Riesgo crítico. Intervención inmediata necesaria."

def factores(d):
    f=[]
    if d.antiguedad_meses<3:  f.append(f"Trabajador nuevo ({d.antiguedad_meses} meses) — etapa más vulnerable")
    elif d.antiguedad_meses<6: f.append(f"Antigüedad baja ({d.antiguedad_meses} meses)")
    if d.cumple_epp==0:        f.append("No usa EPP correctamente")
    if d.n_hall_criticos_mes>=2: f.append(f"{d.n_hall_criticos_mes} hallazgos críticos en el área")
    if d.n_actos_sub_trab_mes>=2: f.append(f"{d.n_actos_sub_trab_mes} actos sub-estándar observados")
    if d.cap_reciente_30d==0:  f.append("Sin capacitación SST en los últimos 30 días")
    if d.n_acc_previos_12m>0:  f.append(f"{d.n_acc_previos_12m} accidente(s) en los últimos 12 meses")
    if d.horas_extra_mes>20:   f.append(f"{d.horas_extra_mes} horas extra — riesgo de fatiga")
    if d.turno=="Noche":       f.append("Turno nocturno — mayor riesgo por fatiga")
    if d.nivel_riesgo_puesto=="Alto": f.append("Puesto de riesgo ALTO según IPERC")
    return f[:5] if f else ["Sin factores críticos identificados"]

def recomendaciones(d, p):
    r=[]
    if d.antiguedad_meses<6:    r.append("Asignar mentor o supervisor directo este mes")
    if d.cumple_epp==0:         r.append("Verificar EPP antes de iniciar actividades del día")
    if d.cap_reciente_30d==0:   r.append("Programar capacitación SST esta semana")
    if d.n_hall_criticos_mes>=2: r.append("Levantar hallazgos críticos antes de continuar")
    if d.n_actos_sub_trab_mes>=2: r.append("Realizar conversación de seguridad con el trabajador")
    if d.horas_extra_mes>20:    r.append("Evaluar rotación para reducir horas extra")
    if p>=0.65:
        r.append("Notificar al área de SST — intervención inmediata")
        r.append("Evaluar restricción temporal de tareas de alto riesgo")
    return r[:4] if r else ["Mantener controles actuales y monitoreo mensual"]

@app.get("/health")
def health():
    return {"status":"ok","modelo_cargado":modelo is not None,
            "version":"1.0.0","timestamp":datetime.now().isoformat()}

@app.post("/predecir")
def predecir(d: DatosTrabajador):
    if modelo and features:
        mapa = {"antiguedad_meses":d.antiguedad_meses,"edad":d.edad,
                "horas_trabajadas_mes":d.horas_trabajadas_mes,"horas_extra_mes":d.horas_extra_mes,
                "n_cap_ult_12m":d.n_cap_ult_12m,"cap_reciente_30d":d.cap_reciente_30d,
                "induccion_sst":d.induccion_sst,"cumple_epp":d.cumple_epp,
                "supervision_frecuente":d.supervision_frecuente,
                "n_hall_criticos_mes":d.n_hall_criticos_mes,
                "n_actos_sub_trab_mes":d.n_actos_sub_trab_mes,
                "n_cond_sub_area_mes":d.n_cond_sub_area_mes,
                "n_inc_previos_12m":d.n_inc_previos_12m,"n_acc_previos_12m":d.n_acc_previos_12m,
                "dias_ult_incidente":d.dias_ult_incidente,"dias_ult_accidente":d.dias_ult_accidente,
                "unidad_negocio_enc":UNIDAD_MAP.get(d.unidad_negocio,0),
                "nivel_riesgo_puesto_enc":RIESGO_MAP.get(d.nivel_riesgo_puesto,1),
                "turno_enc":TURNO_MAP.get(d.turno,0)}
        X = np.array([mapa.get(f,0) for f in features]).reshape(1,-1)
        prob = float(modelo.predict_proba(X)[0][1])
    else:
        prob = prob_demo(d)

    nivel, color, msg = nivel_y_color(prob)
    id_eval = str(uuid.uuid4())[:8].upper()
    historial.append({"id_evaluacion":id_eval,"id_trabajador":d.id_trabajador,
                       "unidad_negocio":d.unidad_negocio,"area":d.area,"cargo":d.cargo,
                       "probabilidad_pct":round(prob*100,1),"nivel_riesgo":nivel,
                       "fecha":datetime.now().isoformat(),"supervisor_id":d.supervisor_id})
    return {
        "id_evaluacion":    id_eval,
        "id_trabajador":    d.id_trabajador,
        "unidad_negocio":   d.unidad_negocio,
        "area":             d.area,
        "fecha_evaluacion": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "probabilidad_pct": round(prob*100, 1),
        "nivel_riesgo":     nivel,
        "color_semaforo":   color,
        "factores_riesgo":  factores(d),
        "recomendaciones":  recomendaciones(d, prob),
        "mensaje_resumen":  msg,
    }

@app.get("/historial/{unidad_negocio}")
def get_historial(unidad_negocio: str, limite: int = 20):
    f = [h for h in historial if h["unidad_negocio"].lower()==unidad_negocio.lower()]
    return {"unidad_negocio":unidad_negocio,"total":len(f),"evaluaciones":f[-limite:][::-1]}

@app.get("/dashboard/{unidad_negocio}")
def dashboard(unidad_negocio: str):
    f = [h for h in historial if h["unidad_negocio"].lower()==unidad_negocio.lower()]
    if not f: return {"unidad_negocio":unidad_negocio,"total_evaluaciones":0,"mensaje":"Sin evaluaciones aún"}
    probs=[h["probabilidad_pct"] for h in f]; nivs=[h["nivel_riesgo"] for h in f]
    return {"unidad_negocio":unidad_negocio,"total_evaluaciones":len(f),
            "riesgo_promedio_pct":round(sum(probs)/len(probs),1),"riesgo_maximo_pct":round(max(probs),1),
            "criticos":nivs.count("CRÍTICO"),"altos":nivs.count("ALTO"),
            "medios":nivs.count("MEDIO"),"bajos":nivs.count("BAJO")}

@app.get("/dashboard-general")
def dashboard_general():
    if not historial: return {"mensaje":"Sin datos aún","total":0}
    from collections import Counter
    probs=[h["probabilidad_pct"] for h in historial]
    return {"total_evaluaciones":len(historial),
            "riesgo_promedio_corporacion":round(sum(probs)/len(probs),1),
            "criticos":sum(1 for h in historial if h["nivel_riesgo"]=="CRÍTICO"),
            "altos":sum(1 for h in historial if h["nivel_riesgo"]=="ALTO"),
            "evaluaciones_por_empresa":dict(Counter(h["unidad_negocio"] for h in historial)),
            "ultima_actualizacion":datetime.now().strftime("%d/%m/%Y %H:%M")}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
