import os
import re
import json
import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────
RUTA_BASE   = Path(__file__).parent
RUTA_CONFIG = RUTA_BASE / "config_chatbot.json"

with open(RUTA_CONFIG, encoding="utf-8") as f:
    CFG = json.load(f)


def cargar_api_key(carpeta: Path) -> str:
    """Busca la clave con varios nombres de archivo posibles, en la carpeta de la
    app y en las superiores. Tolera 'CLAVE=valor' o la clave suelta, con comillas."""
    nombres = [".env", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY.env",
               "anthropic_api_key.env", ".env.txt"]
    for base in [carpeta] + list(carpeta.parents)[:3]:
        for nombre in nombres:
            ruta = base / nombre
            if not ruta.is_file():
                continue
            crudo = ruta.read_text(encoding="utf-8", errors="ignore")
            for linea in crudo.splitlines():
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                valor = linea.split("=", 1)[1] if "=" in linea else linea
                clave = valor.strip().strip('"').strip("'").strip()
                if clave.startswith("sk-ant"):
                    return clave
    load_dotenv()
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"].strip()
except Exception:
    ANTHROPIC_API_KEY = cargar_api_key(RUTA_BASE)
def _resolver_ruta(valor, respaldo_relativo):
    p = Path(valor)
    if p.is_absolute() and p.exists():
        return str(p)
    rel = (RUTA_BASE / valor)
    if rel.exists():
        return str(rel)
    return str(RUTA_BASE / respaldo_relativo)

RUTA_DB     = _resolver_ruta(CFG["RUTA_DB"],     "datos/mdpet.db")
RUTA_CHROMA = _resolver_ruta(CFG["RUTA_CHROMA"], "datos/chromadb")
MODELO_CLASIFICADOR = CFG["MODELO_CLASIFICADOR"]
MODELO_SQL          = CFG.get("MODELO_SQL", CFG["MODELO_CLASIFICADOR"])
MODELO_RESPUESTA    = CFG["MODELO_RESPUESTA"]
MODELO_EMBEDDINGS   = CFG["MODELO_EMBEDDINGS"]
COLECCION_PDF       = CFG["COLECCION_PDF"]
UMBRAL_CHROMA       = CFG["UMBRAL_CHROMA"]
N_RESULTADOS_CHROMA = CFG["N_RESULTADOS_CHROMA"]

if not ANTHROPIC_API_KEY:
    archivos = [p.name for p in RUTA_BASE.iterdir() if p.is_file()]
    st.error(
        f"No encontré la clave de Anthropic.\n\n"
        f"Carpeta de la app: `{RUTA_BASE}`\n\n"
        f"Archivos en esa carpeta: {archivos}\n\n"
        "Creá ahí un archivo `.env` con una línea:\n\n"
        "`ANTHROPIC_API_KEY=sk-ant-api03-...tu-clave-completa...`"
    )
    st.stop()

if len(ANTHROPIC_API_KEY) < 80:
    st.warning(f"La clave tiene solo {len(ANTHROPIC_API_KEY)} caracteres; "
               "una clave completa supera los 100. Puede estar cortada.")

# ─────────────────────────────────────────────
# Inicialización de recursos (cacheados)
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_recursos():
    claude        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    chroma        = chromadb.PersistentClient(path=RUTA_CHROMA)
    coleccion_pdf = chroma.get_or_create_collection(
        name=COLECCION_PDF, metadata={"hnsw:space": "cosine"}
    )
    modelo_emb = SentenceTransformer(MODELO_EMBEDDINGS)
    conn_db    = sqlite3.connect(RUTA_DB, check_same_thread=False)
    conn_db.row_factory = sqlite3.Row
    return claude, coleccion_pdf, modelo_emb, conn_db

claude, coleccion_pdf, modelo_emb, conn_db = cargar_recursos()

# Rango de años REAL de la base (se adapta solo cuando se agregan años nuevos).
try:
    _rango = conn_db.execute(
        "SELECT MIN(anio), MAX(anio) FROM mediciones").fetchone()
    ANIO_MIN = int(_rango[0]) if _rango and _rango[0] else 2014
    ANIO_MAX = int(_rango[1]) if _rango and _rango[1] else 2025
except Exception:
    ANIO_MIN, ANIO_MAX = 2014, 2025

# ─────────────────────────────────────────────
# Schema SQL construido con los temas reales de la base
# ─────────────────────────────────────────────
@st.cache_data
def construir_schema():
    try:
        temas = [r[0] for r in conn_db.execute(
            "SELECT tema FROM meta_temas ORDER BY n_indicadores DESC"
        ).fetchall()]
    except sqlite3.OperationalError:
        temas = [r[0] for r in conn_db.execute(
            "SELECT DISTINCT tema FROM indicadores ORDER BY tema"
        ).fetchall()]
    lista = "\n".join(f"  - {t}" for t in temas)
    return f"""
Base SQLite con indicadores urbanos de Mar del Plata (2014-2024).

TABLAS:
indicadores (id_indicador, nombre, tema, origen, ultima_medicion, ods, formula, unidad)
mediciones  (id, id_indicador, anio, valor_numerico, valor_texto, es_formula_no_resuelta)
umbrales    (id, id_indicador, color, rango_texto, limite_inferior, limite_superior)

TEMAS DISPONIBLES (usar EXACTAMENTE estos nombres en WHERE i.tema = '...'):
{lista}

NOTAS:
- Siempre hacer JOIN entre mediciones e indicadores.
- Incluir SIEMPRE i.nombre en el SELECT.
- Los valores NULL en valor_numerico significan N/D.
- Para el más reciente: WHERE valor_numerico IS NOT NULL ORDER BY anio DESC LIMIT 1.
"""

SCHEMA_DESCRIPCION = construir_schema()

# ─────────────────────────────────────────────
# Capa de respuesta directa sin LLM (Nivel 0)
# ─────────────────────────────────────────────
import re
import unicodedata
from difflib import SequenceMatcher

STOPWORDS = {
    "cual", "cuales", "fue", "es", "son", "el", "la", "los", "las", "de", "del",
    "en", "para", "por", "un", "una", "y", "o", "a", "al", "con", "que", "se",
    "cuanto", "cuanta", "cuantos", "cuantas", "valor", "dato", "dame", "decime",
    "mostrame", "quiero", "saber", "sobre", "mar", "plata", "mdp", "ciudad",
    "anio", "ano", "hubo", "habia", "tiene", "tenia", "me", "podes", "hay",
    "anos", "indicador", "indicadores", "actual", "actualmente", "ultimo",
    "ultima", "reciente", "mas", "esta", "estuvo", "tuvo",
}


def normalizar(texto: str) -> str:
    t = texto.lower().strip()
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens_utiles(texto: str) -> set:
    return {w for w in normalizar(texto).split() if w not in STOPWORDS and len(w) > 2}


def extraer_anios(pregunta: str) -> list:
    """Años mencionados en la pregunta que están dentro del rango real de la base."""
    return sorted({int(a) for a in re.findall(r"\b(20\d{2})\b", pregunta)
                   if ANIO_MIN <= int(a) <= ANIO_MAX})


def hay_anio_fuera_de_rango(pregunta: str) -> bool:
    """True si menciona un año 20xx que NO está en el rango real de la base."""
    todos = [int(a) for a in re.findall(r"\b(20\d{2})\b", pregunta)]
    return any(not (ANIO_MIN <= a <= ANIO_MAX) for a in todos)


_PALABRAS_EVOLUCION = [
    "evolución", "evolucion", "evolucionó", "evoluciono", "evolucionar",
    "tendencia", "cómo cambió", "como cambio", "cómo varió", "como vario",
    "a lo largo", "con el tiempo", "en el tiempo", "a través de los años",
    "mejoró", "mejoro", "empeoró", "empeoro", "subió", "subio", "bajó", "bajo",
    "aumentó", "aumento", "disminuyó", "disminuyo", "creció", "crecio",
    "evolucion", "serie", "histórico", "historico", "trayectoria",
]


def es_pregunta_evolucion(pregunta: str) -> bool:
    """True si la pregunta pide una tendencia/evolución (y no un dato puntual)."""
    p = normalizar(pregunta)
    return any(normalizar(w) in p for w in _PALABRAS_EVOLUCION)


class MatcherDirecto:
    """Responde sin LLM cuando identifica un indicador con alta confianza."""

    UMBRAL_GANADOR = 0.55   # score mínimo para aceptar un match
    MARGEN_CLARO   = 0.15   # ventaja mínima sobre el 2do para responder solo

    def __init__(self, conn):
        self.conn = conn
        # Detectar si existe la columna 'formato' (base creada por el 01 nuevo)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(indicadores)").fetchall()]
        campo_formato = "formato" if "formato" in cols else "'numero' AS formato"
        self.indicadores = [
            {"id": r[0], "nombre": r[1], "tema": r[2], "formato": r[3],
             "tokens": tokens_utiles(r[1]), "norm": normalizar(r[1])}
            for r in conn.execute(
                f"SELECT id_indicador, nombre, tema, {campo_formato} FROM indicadores"
            ).fetchall()
        ]

    # ── scoring ──────────────────────────────────────────────
    def _score(self, tks_preg, norm_preg, ind):
        if not tks_preg or not ind["tokens"]:
            return 0.0
        inter     = tks_preg & ind["tokens"]
        cobertura = len(inter) / len(ind["tokens"])
        precision = len(inter) / len(tks_preg)
        bonus     = 0.25 if ind["norm"] in norm_preg else 0.0
        fuzzy     = SequenceMatcher(None, norm_preg, ind["norm"]).ratio() * 0.15
        return min(1.0, 0.55 * cobertura + 0.30 * precision + bonus + fuzzy)

    def buscar(self, pregunta: str):
        tks  = tokens_utiles(pregunta)
        norm = normalizar(pregunta)
        return sorted(((self._score(tks, norm, i), i) for i in self.indicadores),
                      key=lambda x: -x[0])

    # ── formateo ─────────────────────────────────────────────
    def _num(self, v: float) -> str:
        """Formato es-AR: miles con punto, decimales con coma."""
        s = f"{v:,.2f}"
        if s.endswith(".00"):
            s = s[:-3]
        return s.replace(",", "@").replace(".", ",").replace("@", ".")

    def _fmt_valor(self, vn, vt, ind) -> str:
        if vn is None:
            return vt if vt else "sin dato"
        if ind["formato"] == "porcentaje":
            return f"{self._num(vn * 100)}%"
        return self._num(vn)

    # ── resolución ───────────────────────────────────────────
    def resolver(self, pregunta: str):
        """Retorna dict con la respuesta lista, o None si no está seguro."""
        anios     = extraer_anios(pregunta)
        puntuados = self.buscar(pregunta)
        if not puntuados:
            return None

        mejor_score, mejor = puntuados[0]
        segundo_score      = puntuados[1][0] if len(puntuados) > 1 else 0.0

        if mejor_score < self.UMBRAL_GANADOR:
            return None                      # no reconozco el indicador
        if (mejor_score - segundo_score) < self.MARGEN_CLARO:
            return None                      # ambiguo entre varios indicadores

        serie = self.conn.execute(
            """SELECT anio, valor_numerico, valor_texto FROM mediciones
               WHERE id_indicador=?
                 AND (valor_numerico IS NOT NULL OR valor_texto IS NOT NULL)
               ORDER BY anio""", (mejor["id"],)).fetchall()
        if not serie:
            return None

        d   = {a: (vn, vt) for a, vn, vt in serie}
        pie = "\n\n_Fuente: base de indicadores MDPET (planilla oficial)._"

        # Si pidió un año fuera de 2014-2024, avisar en vez de dar otro dato
        if hay_anio_fuera_de_rango(pregunta) and not anios:
            disp = ", ".join(str(x) for x in sorted(d))
            txt  = (f"El indicador **{mejor['nombre']}** ({mejor['tema']}) solo tiene datos "
                    f"entre {ANIO_MIN} y {ANIO_MAX}.\n\nAños disponibles: {disp}.")
            return {"respuesta": txt + pie, "indicador": mejor["nombre"], "tema": mejor["tema"],
                    "via": "directo_fuera_rango", "score": round(mejor_score, 3)}

        # ── Evolución / tendencia
        #    Se dispara si pidieron un rango de 2+ años, O si la pregunta usa
        #    palabras de evolución (aunque mencionen 0 o 1 año: "cómo evolucionó").
        pide_evolucion = es_pregunta_evolucion(pregunta)
        if len(anios) >= 2 or (pide_evolucion and len(d) >= 2):
            if len(anios) >= 2:
                a0, a1 = anios[0], anios[-1]
            else:
                a0, a1 = min(d), max(d)   # toda la serie disponible
            tramo  = [(a, d[a]) for a in sorted(d) if a0 <= a <= a1]
            if not tramo:
                return None
            lineas = [f"- {a}: {self._fmt_valor(v[0], v[1], mejor)}" for a, v in tramo]
            serie_txt = "\n".join(lineas)
            prim, ult = tramo[0][1][0], tramo[-1][1][0]
            resumen = ""
            if prim not in (None, 0) and ult is not None:
                var   = (ult - prim) / abs(prim) * 100
                signo = "aumentó" if var > 0 else "disminuyó"
                resumen = (f"Entre {tramo[0][0]} y {tramo[-1][0]} {signo} "
                           f"{abs(var):.1f}% en términos relativos.")
            txt = (f"**{mejor['nombre']}** ({mejor['tema']}), evolución {a0}–{a1}:\n\n"
                   + serie_txt + (f"\n\n{resumen}" if resumen else ""))
            # Devolvemos la serie para que responder() la pase al LLM a interpretar.
            return {"respuesta": txt + pie, "indicador": mejor["nombre"], "tema": mejor["tema"],
                    "via": "directo_evolucion", "score": round(mejor_score, 3),
                    "interpretar": True, "serie": serie_txt, "resumen": resumen,
                    "nombre_ind": mejor["nombre"]}

        # ── Año puntual
        if len(anios) == 1:
            a = anios[0]
            if a not in d:
                disp = ", ".join(str(x) for x in sorted(d))
                txt  = (f"No hay dato de **{mejor['nombre']}** ({mejor['tema']}) para {a}.\n\n"
                        f"Años con dato disponible: {disp}.\n\n"
                        f"Para informes y documentos completos: "
                        f"https://mardelplataentretodos.org/repositorio/")
                return {"respuesta": txt + pie, "indicador": mejor["nombre"], "tema": mejor["tema"],
                        "via": "directo_sin_dato", "score": round(mejor_score, 3)}
            vn, vt = d[a]
            txt = (f"**{mejor['nombre']}** ({mejor['tema']}) en {a}: "
                   f"**{self._fmt_valor(vn, vt, mejor)}**")
            return {"respuesta": txt + pie, "indicador": mejor["nombre"], "tema": mejor["tema"],
                    "via": "directo_anio", "score": round(mejor_score, 3)}

        # ── Sin año: último disponible
        a_ult  = max(d)
        vn, vt = d[a_ult]
        txt = (f"**{mejor['nombre']}** ({mejor['tema']}), último dato disponible "
               f"({a_ult}): **{self._fmt_valor(vn, vt, mejor)}**")
        return {"respuesta": txt + pie, "indicador": mejor["nombre"], "tema": mejor["tema"],
                "via": "directo_ultimo", "score": round(mejor_score, 3)}

@st.cache_resource
def cargar_matcher():
    return MatcherDirecto(conn_db)

matcher = cargar_matcher()


# ─────────────────────────────────────────────
# Motor RAG (Anthropic)
# ─────────────────────────────────────────────
def llamar_claude(system: str, user: str, modelo: str, max_tokens: int = 800,
                  historial: list | None = None) -> str:
    """Llama a Claude. Si se pasa 'historial' (lista de turnos previos
    {'role': 'user'|'assistant', 'content': str}), se incluye antes del
    mensaje actual para que el modelo mantenga el hilo de la conversación."""
    mensajes = []
    if historial:
        mensajes.extend(historial)
    mensajes.append({"role": "user", "content": user})
    resp = claude.messages.create(
        model=modelo, max_tokens=max_tokens, system=system,
        messages=mensajes,
    )
    return resp.content[0].text.strip()


# Cuántos turnos (pregunta+respuesta) de contexto se le pasan a Claude.
# Mantenerlo bajo cuida el costo: 3 turnos alcanzan para seguir el hilo.
MAX_TURNOS_MEMORIA = 3


def construir_historial(mensajes: list) -> list:
    """Toma st.session_state.mensajes y arma el historial para Claude:
    los últimos MAX_TURNOS_MEMORIA intercambios, sin el mensaje actual.
    Trunca cada respuesta larga para no inflar el costo de tokens."""
    hist = []
    for m in mensajes:
        rol = "user" if m.get("rol") == "usuario" else "assistant"
        texto = (m.get("contenido") or "").strip()
        if not texto:
            continue
        # Recortar respuestas muy largas (series, etc.) para ahorrar tokens.
        if len(texto) > 600:
            texto = texto[:600] + " […]"
        hist.append({"role": rol, "content": texto})
    # Nos quedamos con los últimos N turnos (2 mensajes = 1 turno).
    return hist[-(MAX_TURNOS_MEMORIA * 2):]


PALABRAS_SQL = ["cuánto", "cuanto", "cuál fue", "cual fue", "valor", "porcentaje",
                "tasa", "cantidad", "evolución", "evolucion", "comparar", "entre 20",
                "en 20", "promedio", "más reciente", "mas reciente", "último", "ultimo"]
PALABRAS_CHROMA = ["metodología", "metodologia", "cómo se mide", "como se mide",
                   "qué significa", "que significa", "explicá", "explica", "por qué",
                   "por que", "definición", "definicion", "informe", "documento"]
PALABRAS_FUERA = ["pizzería", "pizzeria", "playa", "restaurante", "hotel", "clima",
                  "receta", "chiste", "fútbol", "futbol"]


def clasificar_por_reglas(pregunta: str):
    p = pregunta.lower()
    if any(w in p for w in PALABRAS_FUERA):
        return {"tipo": "fuera_alcance", "razon": "regla"}
    sql = any(w in p for w in PALABRAS_SQL)
    chr = any(w in p for w in PALABRAS_CHROMA)
    if sql and chr: return {"tipo": "ambos", "razon": "regla"}
    if sql:  return {"tipo": "sql", "razon": "regla"}
    if chr:  return {"tipo": "chroma", "razon": "regla"}
    return None


def clasificar_pregunta(pregunta: str) -> dict:
    regla = clasificar_por_reglas(pregunta)
    if regla is not None:
        return regla
    system = ('Clasificá la pregunta y respondé SOLO con JSON: '
              '{"tipo": "sql"|"chroma"|"ambos"|"fuera_alcance", "razon": "breve"}. '
              'sql=dato numérico; chroma=contexto PDF; ambos=dato+contexto; '
              'fuera_alcance=no es sobre indicadores urbanos de Mar del Plata.')
    texto = llamar_claude(system, pregunta, MODELO_CLASIFICADOR, 150)
    try:
        match = re.search(r"\{.*?\}", texto, re.DOTALL)
        return json.loads(match.group()) if match else {"tipo": "ambos", "razon": "fallback"}
    except Exception:
        return {"tipo": "ambos", "razon": "error"}


PALABRAS_PELIGROSAS = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "ATTACH", "PRAGMA"]


def generar_y_ejecutar_sql(pregunta: str) -> dict:
    system = (f"Generá SOLO el SQL (sin markdown) para SQLite que responda la pregunta.\n"
              f"{SCHEMA_DESCRIPCION}\n"
              "Usá siempre JOIN. Incluí i.nombre. Buscá nombres con LIKE. Terminá con LIMIT 50.")
    sql = llamar_claude(system, pregunta, MODELO_SQL, 400)
    # Limpiar markdown, comentarios y el ; final (aunque tenga espacios/saltos)
    sql = re.sub(r"```sql|```", "", sql)
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = sql.strip().rstrip(";").strip()
    if "limit" not in sql.lower():
        sql = sql + " LIMIT 50"
    # Detectar ; reales, ignorando los que están dentro de strings
    sin_strings = re.sub(r"'[^']*'", "''", sql)
    sin_strings = re.sub(r'"[^"]*"', '""', sin_strings)
    if ";" in sin_strings:
        return {"sql": sql, "filas": [], "error": "múltiples sentencias"}
    if not sql.lstrip().lower().startswith(("select", "with")):
        return {"sql": sql, "filas": [], "error": "solo se permiten consultas SELECT"}
    for p in PALABRAS_PELIGROSAS:
        if re.search(rf"\b{p}\b", sql, re.IGNORECASE):
            return {"sql": sql, "filas": [], "error": f"contiene {p}"}
    try:
        cur      = conn_db.execute(sql)
        columnas = [d[0] for d in cur.description]
        filas    = [dict(zip(columnas, row)) for row in cur.fetchall()]
        return {"sql": sql, "filas": filas, "error": None}
    except sqlite3.Error as e:
        return {"sql": sql, "filas": [], "error": str(e)}


def buscar_chroma(pregunta: str) -> dict:
    if coleccion_pdf.count() == 0:
        return {"chunks": [], "hay_resultado": False}
    embedding = modelo_emb.encode([pregunta]).tolist()
    n_safe    = min(N_RESULTADOS_CHROMA, coleccion_pdf.count())
    resultados = coleccion_pdf.query(
        query_embeddings=embedding, n_results=n_safe,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(resultados["documents"][0],
                               resultados["metadatas"][0],
                               resultados["distances"][0]):
        if (1 - dist) >= UMBRAL_CHROMA:
            chunks.append({
                "texto": doc, "score": round(1 - dist, 3),
                "titulo": meta.get("titulo", ""), "tema": meta.get("tema", ""),
                "pagina": meta.get("pagina", ""), "fecha": meta.get("fecha", ""),
                "url": meta.get("url_item", ""),
            })
    return {"chunks": chunks, "hay_resultado": len(chunks) > 0}


URL_REPOSITORIO = "https://mardelplataentretodos.org/repositorio/"

SYSTEM_RESPUESTA = (
    "Sos el asistente de Mar del Plata Entre Todos (mardelplataentretodos.org), "
    "plataforma de monitoreo ciudadano de indicadores urbanos, ambientales y fiscales.\n"
    "REGLAS:\n"
    "1. Respondé SOLO con el contexto provisto. Nunca inventes datos.\n"
    "2. Si hay datos numéricos, mostralos antes de cualquier interpretación.\n"
    "3. Citá siempre la fuente (informe, año, página si aplica).\n"
    "4. Si el contexto trae la URL de un item del repositorio, incluila como enlace.\n"
    "5. Si no encontrás el dato, o solo tenés información parcial o desactualizada, "
    "decilo con claridad y remití SIEMPRE al repositorio con este enlace exacto: "
    f"{URL_REPOSITORIO} . Nunca derives a la home del sitio ni inventes otras URLs.\n"
    "6. Priorizá los datos de la base (Excel) sobre los PDF: si la base tiene un valor "
    "más reciente que el PDF, usá el de la base y aclaralo.\n"
    "7. No extrapoles tendencias ni completes años faltantes.\n"
    "8. Español rioplatense, claro y conciso."
)

# System prompt para redactar en tono conversacional los datos que vienen de la
# base (capa directa). La REGLA DE ORO: el número/dato ya está calculado y es el
# que hay que usar tal cual; Claude solo lo envuelve en una frase natural y
# atiende el hilo de la conversación. Nunca inventa ni cambia cifras.
SYSTEM_CONVERSA = (
    "Sos el asistente conversacional de Mar del Plata Entre Todos "
    "(mardelplataentretodos.org), una plataforma ciudadana de indicadores "
    "urbanos, ambientales y fiscales de Mar del Plata.\n"
    "Te voy a pasar un DATO YA VERIFICADO extraído de la base oficial. Tu tarea "
    "es responderle a la persona de forma natural, cálida y conversacional, "
    "en español rioplatense, como lo haría una persona que conoce bien el tema.\n"
    "REGLAS ESTRICTAS:\n"
    "1. Usá EXACTAMENTE las cifras, años y nombres del dato provisto. "
    "Nunca inventes, redondees distinto ni agregues números que no estén.\n"
    "2. No agregues datos de otros años, otras ciudades ni comparaciones que no "
    "estén en el dato provisto. Si no está, no lo digas.\n"
    "3. Podés dar una frase breve de contexto o interpretación simple "
    "(si subió/bajó, si es alto/bajo), pero sin inventar causas ni proyecciones.\n"
    "4. Tené en cuenta el hilo de la conversación: si la persona pregunta algo "
    "de seguimiento ('¿y en 2020?', '¿y comparado?'), respondé a eso.\n"
    "5. Sé breve: 2 a 4 oraciones. Nada de listas largas salvo que el dato lo sea.\n"
    "6. Mantené el enlace a la planilla oficial si viene incluido.\n"
    "7. Tono amable y claro, sin tecnicismos innecesarios."
)


# ============================================================
#  NAVEGACIÓN DEL REPOSITORIO — Colección 9 "Indicadores y sets de datos"
# ============================================================
# Los 15 Excel de "Resumen de Indicadores" por tema, con su link a la
# página del documento en el repositorio (url_item, /items/show/NNN).
# Se prioriza esta colección: es la fuente primaria de los datos numéricos.
#
# Para actualizar: si agregan un Excel nuevo a la colección 9, sumá una
# fila acá con su id_item (el número que aparece en la URL del documento).

BASE_REPO = "https://mardelplataentretodos.org/repositorio"
COLECCION_9_URL = f"{BASE_REPO}/items/browse?collection=9"

# Excel oficiales de indicadores por tema (colección 9).
# Clave: nombre corto para mostrar. Valores: id_item y período.
EXCEL_INDICADORES = [
    {"tema": "Agua, saneamiento y drenaje", "id_item": 149, "periodo": "2014-2024"},
    {"tema": "Ambiente",                    "id_item": 150, "periodo": "2014-2024"},
    {"tema": "Conectividad",                "id_item": 207, "periodo": "2014-2025"},
    {"tema": "Educación",                   "id_item": 153, "periodo": "2014-2024"},
    {"tema": "Economía y Empleo",           "id_item": 152, "periodo": "2014-2025"},
    {"tema": "Energía",                     "id_item": 154, "periodo": "2014-2025"},
    {"tema": "Gestión de Residuos Sólidos", "id_item": 157, "periodo": "2014-2025"},
    {"tema": "Gestión Pública Participativa","id_item": 162, "periodo": "2014-2025"},
    {"tema": "Inequidad Urbana",            "id_item": 155, "periodo": "2014-2025"},
    {"tema": "Logística",                   "id_item": 147, "periodo": "2014-2024"},
    {"tema": "Movilidad",                   "id_item": 156, "periodo": "2014-2025"},
    {"tema": "Salud",                       "id_item": 158, "periodo": "2014-2024"},
    {"tema": "Seguridad Ciudadana",         "id_item": 148, "periodo": "2014-2024"},
    {"tema": "Turismo Sostenible",          "id_item": 159, "periodo": "2014-2024"},
    {"tema": "Vivienda y Uso de Suelo",     "id_item": 160, "periodo": "2014-2025"},
]

for _e in EXCEL_INDICADORES:
    _e["url"] = f"{BASE_REPO}/items/show/{_e['id_item']}"
    _e["titulo"] = f"Resumen de Indicadores de {_e['tema']} ({_e['periodo']})"

# Puente: tema de un indicador en la base  ->  tema del Excel de la colección 9.
# Los temas de la base (28) se mapean al Excel temático que les corresponde.
# Si un tema no está acá, no se ofrece Excel (pero igual se responde el dato).
MAPA_TEMA_A_EXCEL = {
    "Agua": "Agua, saneamiento y drenaje",
    "Saneamiento y Drenaje": "Agua, saneamiento y drenaje",
    "Calidad de Aire": "Ambiente",
    "Mitigación del Cambio Climático": "Ambiente",
    "Ruido": "Ambiente",
    "Conectividad": "Conectividad",
    "Educación": "Educación",
    "Empleo": "Economía y Empleo",
    "Competitividad de la Economía": "Economía y Empleo",
    "Competitividad de la Economía. Logística": "Logística",
    "Competitividad de la Economía. Producción": "Economía y Empleo",
    "Energía": "Energía",
    "Gestión de Residuos Sólidos": "Gestión de Residuos Sólidos",
    "Gestión Pública Participativa": "Gestión Pública Participativa",
    "Gestión Pública Moderna": "Gestión Pública Participativa",
    "Transparencia": "Gestión Pública Participativa",
    "Inequidad Urbana": "Inequidad Urbana",
    "Población": "Inequidad Urbana",
    "Movilidad": "Movilidad",
    "Salud": "Salud",
    "Seguridad Ciudadana": "Seguridad Ciudadana",
    "Turismo Sostenible": "Turismo Sostenible",
    "Vivienda": "Vivienda y Uso de Suelo",
    "Uso del suelo": "Vivienda y Uso de Suelo",
    # Temas de deuda/gasto/impuestos no tienen Excel propio en la colección 9
}

_EXCEL_POR_TEMA = {e["tema"]: e for e in EXCEL_INDICADORES}


def excel_para_tema(tema_base: str):
    """Dado el tema de un indicador de la base, devuelve el Excel oficial
    de la colección 9 que le corresponde, o None si no hay."""
    tema_excel = MAPA_TEMA_A_EXCEL.get(tema_base)
    if not tema_excel:
        return None
    return _EXCEL_POR_TEMA.get(tema_excel)


def enlace_excel(tema_base: str) -> str:
    """Devuelve una línea markdown con el enlace al Excel del tema, o ''."""
    e = excel_para_tema(tema_base)
    if not e:
        return ""
    return (f"\n\n📊 **Ver todos los indicadores de {e['tema']}** "
            f"(datos oficiales, {e['periodo']}): [{e['titulo']}]({e['url']})")


def listado_coleccion9_markdown() -> str:
    """Texto markdown con todos los Excel de indicadores, para el modo explorar."""
    lineas = ["Estos son los **resúmenes oficiales de indicadores** por tema "
              "(planillas Excel con toda la serie, fuentes y cálculos):\n"]
    for e in sorted(EXCEL_INDICADORES, key=lambda x: x["tema"]):
        lineas.append(f"- 📊 [{e['tema']} ({e['periodo']})]({e['url']})")
    lineas.append(f"\n🔗 Ver la colección completa en el repositorio: "
                  f"[Indicadores y sets de datos]({COLECCION_9_URL})")
    return "\n".join(lineas)


def responder(pregunta: str, historial: list | None = None) -> dict:
    # 0. Capa directa: el DATO sale de la base (exacto, sin alucinación);
    #    Claude solo lo redacta en tono conversacional y sigue el hilo.
    directo = matcher.resolver(pregunta)
    if directo is not None:
        enlace = enlace_excel(directo.get("tema", ""))

        # 0.a Evolución / tendencia: Claude interpreta la serie completa.
        if directo.get("interpretar"):
            try:
                system = (
                    "Sos el asistente conversacional de Mar del Plata Entre Todos. "
                    "Te paso la serie histórica REAL de un indicador. Redactá una "
                    "lectura clara y natural de su evolución, en español rioplatense: "
                    "tendencia general, cuánto cambió, y los años de subas o bajas "
                    "marcadas si las hay. USÁ SOLO los valores provistos; no inventes "
                    "ni estimes años faltantes. Tené en cuenta el hilo de la charla "
                    "si la persona viene preguntando algo relacionado. "
                    "Sé conciso (4-6 líneas). No repitas toda la lista año por año.")
                user = (f"Indicador: {directo['nombre_ind']}\n\n"
                        f"Serie:\n{directo['serie']}\n\n"
                        f"{directo.get('resumen','')}\n\nPregunta: {pregunta}")
                narracion = llamar_claude(system, user, MODELO_RESPUESTA, 500,
                                          historial=historial)
                respuesta = (f"{directo['respuesta']}\n\n---\n\n{narracion}{enlace}")
                return {"respuesta": respuesta, "tipo": "evolucion",
                        "sql": None, "chunks": [], "error": None}
            except Exception:
                # Si falla el LLM, al menos devolvemos la serie (gratis, sin romper).
                return {"respuesta": directo["respuesta"] + enlace, "tipo": "directo",
                        "sql": None, "chunks": [], "error": None}

        # 0.b Dato puntual / último valor: Claude lo envuelve en una frase natural.
        #     El número ya está calculado en directo["respuesta"]; Claude NO lo cambia.
        try:
            user = (f"DATO VERIFICADO (de la base oficial):\n{directo['respuesta']}"
                    f"{enlace}\n\nPregunta de la persona: {pregunta}\n\n"
                    "Redactá la respuesta conversacional usando ese dato exacto.")
            texto = llamar_claude(SYSTEM_CONVERSA, user, MODELO_RESPUESTA, 400,
                                  historial=historial)
            return {"respuesta": texto, "tipo": "directo",
                    "sql": None, "chunks": [], "error": None}
        except Exception:
            # Fallback: si Claude falla, devolvemos el texto directo (gratis, sin romper).
            return {"respuesta": directo["respuesta"] + enlace, "tipo": "directo",
                    "sql": None, "chunks": [], "error": None}

    clasificacion = clasificar_pregunta(pregunta)
    tipo = clasificacion["tipo"]

    if tipo == "fuera_alcance":
        return {"respuesta": ("Esta pregunta está fuera del alcance del chatbot. "
                              "Solo respondo sobre indicadores urbanos, ambientales y "
                              "fiscales de Mar del Plata."),
                "tipo": tipo, "sql": None, "chunks": [], "error": None}

    res_sql = res_chroma = None
    if tipo in ("sql", "ambos"):
        res_sql = generar_y_ejecutar_sql(pregunta)
    if tipo in ("chroma", "ambos"):
        res_chroma = buscar_chroma(pregunta)

    # Respaldo: si el SQL no trajo filas pero la pregunta menciona un indicador
    # reconocible, recuperamos la serie directo de la base (sin LLM).
    if res_sql is None or not res_sql["filas"]:
        candidatos = matcher.buscar(pregunta)
        if candidatos and candidatos[0][0] >= 0.20:
            ind = candidatos[0][1]
            serie = conn_db.execute(
                """SELECT anio, valor_numerico, valor_texto FROM mediciones
                   WHERE id_indicador=? AND (valor_numerico IS NOT NULL
                                             OR valor_texto IS NOT NULL)
                   ORDER BY anio DESC LIMIT 12""", (ind["id"],)).fetchall()
            if serie:
                res_sql = {"sql": f"[respaldo desde la base: {ind['nombre']}]",
                           "filas": [{"nombre": ind["nombre"], "tema": ind["tema"],
                                      "anio": a,
                                      "valor": matcher._fmt_valor(vn, vt, ind)}
                                     for a, vn, vt in serie],
                           "error": None}

    hay_sql    = bool(res_sql and res_sql["filas"])
    hay_chroma = bool(res_chroma and res_chroma["hay_resultado"])
    if not hay_sql and not hay_chroma:
        return {"respuesta": ("No encontré información suficiente en los documentos "
                              "disponibles para responder eso con certeza.\n\n"
                              "Te sugiero consultar el repositorio de Mar del Plata "
                              "Entre Todos, donde están los informes completos:\n"
                              f"{URL_REPOSITORIO}"),
                "tipo": "sin_datos", "sql": res_sql, "chunks": [], "error": None}

    partes = []
    if res_sql and not res_sql["error"] and res_sql["filas"]:
        partes.append("=== BASE DE INDICADORES (Excel/SQLite) ===")
        for fila in res_sql["filas"][:15]:
            partes.append(str(fila))
    if res_chroma and res_chroma["hay_resultado"]:
        partes.append("=== FRAGMENTOS DE INFORMES PDF ===")
        for c in res_chroma["chunks"]:
            partes.append(f"[{c['titulo']} | pág.{c['pagina']} | "
                          f"score:{c['score']} | URL: {c.get('url','')}]")
            partes.append(c["texto"])
    contexto = "\n".join(partes) if partes else "No se encontró información relevante."

    respuesta_texto = llamar_claude(
        SYSTEM_RESPUESTA,
        f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta}",
        MODELO_RESPUESTA, 800,
        historial=historial,
    )
    return {"respuesta": respuesta_texto, "tipo": tipo, "sql": res_sql,
            "chunks": res_chroma["chunks"] if res_chroma else [],
            "error": res_sql["error"] if res_sql else None}

# ─────────────────────────────────────────────
# Interfaz Streamlit
# ─────────────────────────────────────────────
# ==========================================================
# Preguntas sugeridas (modo guía)
# ==========================================================
# Preguntas sugeridas, agrupadas por tema. Redactadas para coincidir con los
# nombres reales de los indicadores, de modo que la mayoría se resuelvan por la
# capa directa (gratis). El usuario puede igual escribir libremente lo que quiera.
PREGUNTAS_SUGERIDAS = {
    "💼 Empleo": [
        "¿Cuál es la tasa de desempleo más reciente?",
        "¿Cómo evolucionó la tasa de desempleo?",
        "¿Cuánto es el empleo informal como % del empleo total?",
    ],
    "💧 Agua": [
        "¿Cuál es la calidad de agua más reciente?",
        "¿Cuánto es el consumo anual de agua per cápita?",
        "¿Cuánta agua no contabilizada hay?",
    ],
    "🗑️ Residuos": [
        "¿Cuál es la disposición de residuos domiciliarios por habitante?",
        "¿Cómo evolucionó la disposición de residuos domiciliarios?",
    ],
    "🚌 Movilidad": [
        "¿Cuántos automóviles per cápita hay?",
        "¿Cómo evolucionó la cantidad de automóviles per cápita?",
    ],
    "🛡️ Seguridad": [
        "¿Cuántos homicidios cada 100.000 habitantes hay?",
        "¿Qué porcentaje de ciudadanos se sienten seguros?",
    ],
    "🏥 Salud y educación": [
        "¿Cuántos kinesiólogos cada 1.000 hab. hay?",
        "¿Cuál es la ocupación hotelera promedio anual más reciente?",
    ],
    "💰 Economía y finanzas": [
        "¿Cuál es el presupuesto bruto operativo más reciente?",
        "¿Cómo evolucionó el crecimiento de la deuda?",
    ],
    "🌱 Energía y ambiente": [
        "¿Cuál es el consumo anual de energía eléctrica per cápita?",
        "¿Cuánto es el uso de energía de fuentes renovables no convencionales?",
    ],
    "🏨 Turismo": [
        "¿Cuál es la ocupación hotelera promedio anual?",
        "¿Cómo evolucionó el grado de satisfacción de los residentes con la actividad turística?",
    ],
    "🏠 Vivienda": [
        "¿Cuál es el déficit de vivienda cualitativo más reciente?",
    ],
}


st.set_page_config(page_title="Chatbot · Mar del Plata Entre Todos",
                   page_icon="🌊", layout="centered")

st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
    <h2 style="margin:0">🌊 Mar del Plata Entre Todos</h2>
    <p style="color:gray; margin:4px 0 0">Consultá indicadores urbanos, ambientales y fiscales</p>
</div>
""", unsafe_allow_html=True)
st.divider()


def render_detalle(det):
    tipo_label = {
        "sql":    "📊 Datos de la base de indicadores (Excel/SQLite)",
        "chroma": "📄 Búsqueda en informes PDF",
        "ambos":  "📊📄 Base de indicadores + Informes PDF",
        "directo": "⚡ Respuesta directa desde la base (sin IA)",
        "sin_datos": "🔗 Sin datos suficientes — derivado al repositorio",
    }.get(det.get("tipo", ""), "")
    if tipo_label:
        st.caption(tipo_label)
    if det.get("sql") and det["sql"].get("sql"):
        st.code(det["sql"]["sql"], language="sql")
        if det["sql"]["filas"]:
            import pandas as pd
            st.dataframe(pd.DataFrame(det["sql"]["filas"]), use_container_width=True)
        if det["sql"].get("error"):
            st.error(f"Error SQL: {det['sql']['error']}")
    if det.get("chunks"):
        st.markdown("**Fragmentos de PDF usados:**")
        for c in det["chunks"]:
            st.markdown(f"- **{c['titulo']}** | pág. {c['pagina']} | "
                        f"score: {c['score']} | [{c['tema']}]({c['url']})")


if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{
        "rol": "asistente",
        "contenido": ("¡Hola! Soy el asistente de **Mar del Plata Entre Todos**. "
                      "Podés preguntarme sobre indicadores urbanos, ambientales y "
                      "fiscales de Mar del Plata — o elegir una pregunta sugerida más abajo."),
        "detalle": None,
    }]

# Guarda la pregunta elegida por un botón, para procesarla como si se hubiera escrito
if "pregunta_pendiente" not in st.session_state:
    st.session_state.pregunta_pendiente = None
if "tema_abierto" not in st.session_state:
    st.session_state.tema_abierto = None


def procesar(pregunta):
    """Agrega la pregunta y su respuesta a la conversación."""
    # Historial ANTES de agregar la pregunta actual (memoria conversacional).
    historial = construir_historial(st.session_state.mensajes)
    st.session_state.mensajes.append({"rol": "usuario", "contenido": pregunta, "detalle": None})
    resultado = responder(pregunta, historial=historial)
    detalle = {"tipo": resultado["tipo"], "sql": resultado.get("sql"),
               "chunks": resultado.get("chunks", [])}
    st.session_state.mensajes.append({
        "rol": "asistente", "contenido": resultado["respuesta"], "detalle": detalle,
    })


# ── Mostrar la conversación
for msg in st.session_state.mensajes:
    with st.chat_message("user" if msg["rol"] == "usuario" else "assistant"):
        st.markdown(msg["contenido"])
        if msg.get("detalle"):
            with st.expander("🔍 Ver fuentes y detalles técnicos"):
                render_detalle(msg["detalle"])

# ── Panel de preguntas sugeridas (modo guía)
with st.expander("💡 Preguntas sugeridas por tema", expanded=(len(st.session_state.mensajes) <= 1)):
    st.caption("Elegí un tema y tocá una pregunta, o escribí la tuya abajo.")
    temas = list(PREGUNTAS_SUGERIDAS.keys())
    # Botones de temas en filas de 3
    for fila_ini in range(0, len(temas), 3):
        cols = st.columns(3)
        for col, tema in zip(cols, temas[fila_ini:fila_ini + 3]):
            if col.button(tema, key=f"tema_{tema}", use_container_width=True):
                st.session_state.tema_abierto = (
                    None if st.session_state.tema_abierto == tema else tema)
    # Preguntas del tema abierto
    if st.session_state.tema_abierto:
        st.markdown(f"**{st.session_state.tema_abierto}**")
        for i, preg in enumerate(PREGUNTAS_SUGERIDAS[st.session_state.tema_abierto]):
            if st.button("👉 " + preg, key=f"preg_{st.session_state.tema_abierto}_{i}",
                         use_container_width=True):
                st.session_state.pregunta_pendiente = preg
                st.rerun()

# ── Panel para explorar el repositorio (planillas Excel oficiales, colección 9)
with st.expander("📂 Explorar indicadores por tema (planillas oficiales)"):
    st.markdown(listado_coleccion9_markdown())

# ── Entrada de texto libre (siempre disponible)
entrada = st.chat_input("Escribí tu pregunta sobre Mar del Plata...")

# ── Procesar la pregunta (de un botón o escrita)
pregunta = st.session_state.pregunta_pendiente or entrada
if pregunta:
    st.session_state.pregunta_pendiente = None
    with st.spinner("Buscando en la base de datos e informes..."):
        procesar(pregunta)
    st.rerun()

with st.sidebar:
    st.markdown("### 📊 Estado del sistema")
    try:
        n_chunks = coleccion_pdf.count() if coleccion_pdf is not None else 0
    except Exception:
        n_chunks = 0
    n_ind = conn_db.execute("SELECT COUNT(*) FROM indicadores").fetchone()[0]
    n_med = conn_db.execute(
        "SELECT COUNT(*) FROM mediciones WHERE valor_numerico IS NOT NULL").fetchone()[0]
    st.metric("Indicadores en base", n_ind)
    st.metric("Mediciones numéricas", n_med)
    st.metric("Chunks PDF indexados", n_chunks)
    st.divider()
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.mensajes = [{
            "rol": "asistente", "contenido": "¡Hola! ¿En qué puedo ayudarte hoy?", "detalle": None,
        }]
        st.session_state.tema_abierto = None
        st.session_state.pregunta_pendiente = None
        st.rerun()
    st.divider()
    st.caption("Mar del Plata Entre Todos · mardelplataentretodos.org")