# Chatbot MDPET — Mar del Plata Entre Todos

Asistente conversacional sobre los indicadores urbanos, ambientales y
fiscales de Mar del Plata. Publicado con Streamlit.

## Estructura

- `chatbot_mdpet.py` — la aplicación
- `config_chatbot.json` — configuración (modelos, umbrales, rutas)
- `requirements.txt` — dependencias
- `datos/mdpet.db` — base de indicadores (datos numéricos)
- `datos/chromadb/` — base vectorial de los informes PDF
- `.streamlit/config.toml` — tema visual

## La clave de API NO está en el repositorio

La clave de Anthropic se configura aparte, en los *Secrets* de Streamlit Cloud.
Nunca se sube al repositorio (el `.gitignore` la bloquea).

## Correr en local

```bash
streamlit run chatbot_mdpet.py
```

En local usa las bases de `C:\mdpet_data` si existen; si no, las de `datos/`.
