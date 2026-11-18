"""
identificador.py — Chatbot MDPET
Identifica a qué indicador(es) se refiere una pregunta usando el CATÁLOGO COMPLETO
de la base (opción A). El usuario no necesita escribir el nombre literal: el modelo
reconoce nombres incompletos, sinónimos, siglas, errores de tipeo y preguntas
indirectas, y además entiende preguntas de seguimiento ("¿y en 2019?").

Una sola llamada a Claude (el modelo clasificador, Haiku) resuelve a la vez:
  - el tipo de pregunta (sql / chroma / ambos / fuera_alcance)
  - los id_indicador involucrados
  - la pregunta reescrita con el nombre oficial (mejora también la búsqueda en PDF)

El catálogo va en un bloque con caché de prompt: desde la segunda consulta
(dentro de los ~5 minutos) esa parte se cobra a una fracción del precio normal.

Cuando hay ids, los datos se traen directo de SQLite por id (sin generar SQL con
el modelo): es exacto, más rápido y ahorra una llamada a la API.

No hay que tocar este archivo al actualizar la planilla: el catálogo se lee de
mdpet.db al iniciar la app. Si algún día la tabla `indicadores` tiene una columna
`sinonimos`, se incluye sola en el catálogo.
"""
from __future__ import annotations

import json
import re

TIPOS_VALIDOS = {"sql", "chroma", "ambos", "fuera_alcance"}
MAX_IDS = 10

INSTRUCCIONES = """Sos el primer paso del chatbot de Mar del Plata Entre Todos (mardelplataentretodos.org), \
que responde sobre indicadores urbanos, ambientales, sociales y fiscales de Mar del Plata.

Tu tarea con cada pregunta:
1) Clasificarla:
   - "sql": pide valores, datos, evolución, comparaciones o rankings de indicadores
   - "chroma": pide explicación, metodología, contexto o conclusiones de los informes
   - "ambos": necesita datos y también explicación o contexto
   - "fuera_alcance": no tiene relación con Mar del Plata ni con sus indicadores
2) Identificar a qué indicadores del CATÁLOGO se refiere.

Cómo identificar indicadores:
- El usuario casi nunca escribe el nombre exacto. Reconocé nombres incompletos o \
abreviados, sinónimos, siglas, errores de tipeo y preguntas indirectas \
(ej.: "cuánta gente no tiene trabajo" -> la tasa de desempleo).
- Usá SOLO ids que figuren en el catálogo. Nunca inventes ids.
- Si la pregunta es ambigua entre pocos indicadores parecidos (ej.: dos variantes de \
un mismo indicador), incluí todos los candidatos en "ids" y marcá "ambigua": true.
- Si la pregunta es general o abarca un tema entero (ej.: "¿qué indicadores están en \
rojo?", "¿cómo está Salud?"), dejá "ids" vacío.
- Si la pregunta es de seguimiento ("¿y en 2019?", "¿y el otro?"), usá la \
conversación previa para saber de qué indicador se habla.

Respondé ÚNICAMENTE con un JSON, sin texto antes ni después:
{"tipo": "sql|chroma|ambos|fuera_alcance", "ids": [id, ...], "ambigua": false, \
"pregunta_completa": "...", "razon": "..."}

- "pregunta_completa": la pregunta reescrita, autosuficiente, con el nombre oficial \
del indicador y lo que aporte la conversación previa (años, comparaciones).
- "razon": explicación breve (máximo 15 palabras)."""


def _fmt(v) -> str:
    """Formatea un valor numérico sin decimales de sobra."""
    if v is None:
        return "N/D"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v.is_integer():
        return str(int(v))
    return f"{round(v, 2 if abs(v) >= 1 else 4):g}"


class Identificador:
    def __init__(self, conn, cliente, modelo: str, max_historial: int = 6, formatear=None):
        """
        conn      : conexión sqlite3 a mdpet.db
        cliente   : anthropic.Anthropic(...)
        modelo    : modelo clasificador (ej. el MODELO_CLASIFICADOR de config_chatbot.json)
        formatear : opcional, función (id_indicador, valor_numerico, valor_texto) -> str
                    para mostrar los valores con el formato de la app (%, coma decimal)
        """
        self.conn = conn
        self.cliente = cliente
        self.modelo = modelo
        self.max_historial = max_historial
        self.formatear = formatear
        self.indicadores = self._leer_indicadores()
        self.catalogo_texto = self._armar_catalogo()
        self.ultimo_uso = None  # tokens de la última llamada (para verificar la caché)

    # ── Catálogo ───────────────────────────────────────────────────────────
    def _leer_indicadores(self) -> dict:
        columnas = {r[1] for r in self.conn.execute("PRAGMA table_info(indicadores)")}
        col_sin = "sinonimos" if "sinonimos" in columnas else "NULL"
        filas = self.conn.execute(
            f"SELECT id_indicador, nombre, tema, unidad, {col_sin} "
            "FROM indicadores ORDER BY tema, nombre"
        ).fetchall()
        return {
            int(f[0]): {
                "id": int(f[0]),
                "nombre": str(f[1]).strip(),
                "tema": str(f[2]).strip(),
                "unidad": (f[3] or "").strip(),
                "sinonimos": (f[4] or "").strip(),
            }
            for f in filas
        }

    def _armar_catalogo(self) -> str:
        """Catálogo compacto agrupado por tema: '## Tema' y luego 'id | nombre'."""
        lineas, tema_actual = [], None
        for ind in self.indicadores.values():  # ya vienen ordenados por tema
            if ind["tema"] != tema_actual:
                tema_actual = ind["tema"]
                lineas.append(f"## {tema_actual}")
            linea = f"{ind['id']} | {ind['nombre']}"
            if ind["sinonimos"]:
                linea += f" (también: {ind['sinonimos']})"
            lineas.append(linea)
        return "\n".join(lineas)

    # ── Identificación ─────────────────────────────────────────────────────
    def _texto_historial(self, historial) -> str:
        if not historial:
            return ""
        partes = []
        for m in historial[-self.max_historial:]:
            rol = m.get("rol") or m.get("role") or ""
            contenido = str(m.get("contenido") or m.get("content") or "")
            quien = "Usuario" if rol in ("usuario", "user") else "Asistente"
            partes.append(f"{quien}: {contenido[:400]}")
        return "CONVERSACIÓN PREVIA:\n" + "\n".join(partes) + "\n\n"

    def identificar(self, pregunta: str, historial=None) -> dict:
        """
        Devuelve:
        {"tipo", "ids", "ambigua", "pregunta_completa", "razon", "indicadores", "error"}
        Ante cualquier error devuelve tipo "ambos" sin ids, para que el chatbot
        siga funcionando como antes (generación de SQL + búsqueda en PDF).
        """
        contenido = f"{self._texto_historial(historial)}PREGUNTA ACTUAL: {pregunta}"
        try:
            resp = self.cliente.messages.create(
                model=self.modelo,
                max_tokens=400,
                temperature=0,
                system=[
                    {"type": "text", "text": INSTRUCCIONES},
                    {
                        "type": "text",
                        "text": "CATÁLOGO DE INDICADORES (id | nombre):\n" + self.catalogo_texto,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                messages=[{"role": "user", "content": contenido}],
            )
            self.ultimo_uso = getattr(resp, "usage", None)
            return self._interpretar(resp.content[0].text, pregunta)
        except Exception as e:  # sin conexión, clave inválida, etc.
            return self._resultado("ambos", [], False, pregunta, "fallback", error=str(e))

    def _interpretar(self, texto: str, pregunta: str) -> dict:
        match = re.search(r"\{.*\}", texto or "", re.DOTALL)
        try:
            datos = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            datos = {}
        if not datos:
            return self._resultado("ambos", [], False, pregunta, "respuesta no interpretable")

        tipo = str(datos.get("tipo", "")).strip().lower()
        if tipo not in TIPOS_VALIDOS:
            tipo = "ambos"

        ids = []
        for x in datos.get("ids") or []:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if i in self.indicadores and i not in ids:  # descarta ids inventados
                ids.append(i)
        ids = ids[:MAX_IDS]

        completa = str(datos.get("pregunta_completa") or "").strip() or pregunta
        return self._resultado(
            tipo, ids, bool(datos.get("ambigua")), completa, str(datos.get("razon", ""))
        )

    def _resultado(self, tipo, ids, ambigua, completa, razon, error=None) -> dict:
        return {
            "tipo": tipo,
            "ids": ids,
            "ambigua": ambigua and len(ids) > 1,
            "pregunta_completa": completa,
            "razon": razon,
            "indicadores": [self.indicadores[i]["nombre"] for i in ids],
            "error": error,
        }

    # ── Datos por id ───────────────────────────────────────────────────────
    def datos_por_ids(self, ids: list) -> dict:
        """
        Trae todas las mediciones (y el semáforo) de los indicadores pedidos.
        Devuelve el mismo formato que generar_y_ejecutar_sql: {"sql", "filas", "error"}.
        Una fila por indicador con un año por columna (como la planilla), así el
        contexto queda compacto y la tabla del expander se lee fácil.
        """
        ids = [int(i) for i in ids if int(i) in self.indicadores][:MAX_IDS]
        if not ids:
            return {"sql": None, "filas": [], "error": "Sin indicadores identificados"}
        marcas = ",".join("?" * len(ids))
        try:
            meds = self.conn.execute(
                "SELECT id_indicador, anio, valor_numerico, valor_texto FROM mediciones "
                f"WHERE id_indicador IN ({marcas}) ORDER BY id_indicador, anio",
                ids,
            ).fetchall()
            try:
                umbrales = self.conn.execute(
                    "SELECT id_indicador, color, rango_texto FROM umbrales "
                    f"WHERE id_indicador IN ({marcas})",
                    ids,
                ).fetchall()
            except Exception:
                umbrales = []
        except Exception as e:
            return {"sql": None, "filas": [], "error": str(e)}

        filas = {}
        for i in ids:
            ind = self.indicadores[i]
            filas[i] = {"indicador": ind["nombre"], "tema": ind["tema"], "unidad": ind["unidad"] or "-"}
        for id_ind, anio, v_num, v_txt in meds:
            if v_num is None and not v_txt:
                valor = "N/D"
            elif self.formatear:
                valor = self.formatear(int(id_ind), v_num, v_txt)
            else:
                valor = _fmt(v_num) if v_num is not None else v_txt
            filas[int(id_ind)][str(anio)] = valor
        semaforo = {}
        for id_ind, color, rango in umbrales:
            semaforo.setdefault(int(id_ind), []).append(f"{color}: {rango}")
        for i, partes in semaforo.items():
            filas[i]["semaforo"] = "; ".join(partes)

        consulta = (
            "-- Consulta directa por indicador identificado\n"
            "SELECT * FROM mediciones WHERE id_indicador IN ("
            + ", ".join(map(str, ids)) + ")"
        )
        return {"sql": consulta, "filas": list(filas.values()), "error": None}
