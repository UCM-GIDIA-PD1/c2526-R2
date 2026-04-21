# App web demo para servir modelos

Esta carpeta contiene una aplicacion web demo (FastAPI + frontend estatico) para validar interfaz y endpoints sin depender de modelos conectados.

## Estructura propuesta

```text
c2526-R2/
├── src/                           # Modelos y pipeline existentes
│   └── model_artifacts/           # Artefactos serializados (ejemplo)
│       ├── venta_model.pkl
│       ├── alquiler_model.pkl
│       ├── texto_model.pkl
│       └── imagen_model.pkl
├── app/
│   ├── main.py                    # Entrypoint FastAPI
│   ├── api/
│   │   └── routes.py              # Endpoints de prediccion
│   ├── core/
│   │   └── config.py              # Rutas base y defaults
│   ├── services/
│   │   └── demo_predictors.py     # Predicciones demo desacopladas
│   ├── web/
│   │   ├── index.html             # UI principal
│   │   ├── style.css              # Estilos
│   │   └── app.js                 # Llamadas frontend a la API
│   └── schemas.py                 # Validacion de payloads
├── pyproject.toml
└── Containerfile
```

## Endpoints disponibles

- `POST /predict/venta`
- `POST /predict/alquiler`
- `POST /predict/texto`
- `POST /predict/imagen`
- `GET /health`

## Ejecucion

1. Crear/activar entorno virtual.
2. Instalar dependencias:

```bash
uv sync
```

3. Lanzar la app:

```bash
uv run uvicorn app.main:app --reload
```

4. Abrir:

- Web: <http://127.0.0.1:8000/>
- Docs API: <http://127.0.0.1:8000/docs>

## Ejecucion con Podman

```bash
podman build -t maiday-web-demo -f Containerfile .
podman run --rm -p 8000:8000 maiday-web-demo
```

## Nota

Esta version es una demo funcional. Cuando quieras conectar modelos reales de `src/`, se sustituye `app/services/demo_predictors.py` por servicios de inferencia reales.
