import pandas as pd
from src.utils.funciones_minio import crear_cliente_minio, bajar_minio

def build_fields_and_html(X, form_id, btn_text):
    py_fields = []
    html_fields = ""
    js_parse = ""
    for col in X.columns:
        dtype = X[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            py_type = 'float'
            html_type = 'number" step="any'
            js = f'Number(data.get("{col}"))'
        else:
            py_type = 'str'
            html_type = 'text'
            js = f'String(data.get("{col}"))'
            
        py_fields.append(f'    {col}: {py_type}')
        label_text = col.replace('_', ' ').capitalize()
        html_fields += f'''
        <label>
          {label_text}
          <input name="{col}" type="{html_type}" placeholder="Ej: val" required>
        </label>'''
        js_parse += f'\\n    {col}: {js},'
        
    class_def = '\\n'.join(py_fields)
    form_html = f'<form id="{form_id}" class="form-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">{html_fields}\\n        <button type="submit" style="grid-column: 1 / -1;">{btn_text}</button>\\n      </form>'
    return class_def, form_html, js_parse

def generate_code():
    c = crear_cliente_minio()
    df_v = bajar_minio(c, 'dataset_ml/precios/ventas', 'df_ventas_arboles.parquet')
    df_a = bajar_minio(c, 'dataset_ml/precios/alquiler', 'df_alquiler_arboles.parquet')
    
    X_v = df_v.drop(columns=['Precio'])
    X_a = df_a.drop(columns=['Precio'])
    
    v_class, v_html, v_js = build_fields_and_html(X_v, "venta-form", "Predecir precio de venta")
    a_class, a_html, a_js = build_fields_and_html(X_a, "alquiler-form", "Predecir precio de alquiler")
    
    # 1. Generate schemas.py
    with open('app/schemas.py', 'w') as f:
        f.write(f"from pydantic import BaseModel\\nfrom typing import Any, Optional\\n\\nclass VentaInput(BaseModel):\\n{v_class}\\n\\nclass AlquilerInput(BaseModel):\\n{a_class}\\n\\nclass TextoInput(BaseModel):\\n    texto: str\\n\\nclass PredictionResponse(BaseModel):\\n    model_name: str\\n    prediction: float | str | int\\n")
        
    # 2. Update routes
    routes_code = '''from fastapi import APIRouter
from app.schemas import AlquilerInput, PredictionResponse, TextoInput, VentaInput
from app.services.predictors import predict_tabular

router = APIRouter(prefix="/predict", tags=["predictions"])

@router.post("/venta", response_model=PredictionResponse)
def predict_venta(data: VentaInput) -> PredictionResponse:
    prediction = predict_tabular("venta", data.model_dump())
    return PredictionResponse(model_name="venta-xgboost", prediction=prediction)

@router.post("/alquiler", response_model=PredictionResponse)
def predict_alquiler(data: AlquilerInput) -> PredictionResponse:
    prediction = predict_tabular("alquiler", data.model_dump())
    return PredictionResponse(model_name="alquiler-xgboost", prediction=prediction)
'''
    with open('app/api/routes.py', 'w') as f:
        f.write(routes_code)

    # 3. Read HTML and inject fields
    with open('app/web/precios.html', 'r', encoding='utf-8') as f:
        html = f.read()

    import re
    html = re.sub(r'<form id="venta-form" .*?</form>', v_html, html, flags=re.DOTALL)
    html = re.sub(r'<form id="alquiler-form" .*?</form>', a_html, html, flags=re.DOTALL)

    with open('app/web/precios.html', 'w', encoding='utf-8') as f:
        f.write(html)
        
    # 4. Update JS (a_js has the superset so we can safely use one parser that picks from FormData, but let's just make two parsers or one parser that tries both. Wait! Since FormData.get returns null if missing, we can just use the superset of fields or create specific parsers. To be safe, let's use the superset).
    # Superset of fields:
    all_cols = list(set(X_v.columns) | set(X_a.columns))
    js_parse = ""
    for col in all_cols:
        col_type = df_a[col].dtype if col in df_a.columns else df_v[col].dtype
        js = f'Number(data.get("{col}"))' if pd.api.types.is_numeric_dtype(col_type) else f'String(data.get("{col}"))'
        js_parse += f'\\n    {col}: {js},'
        
    with open('app/web/app.js', 'r', encoding='utf-8') as f:
        app_js = f.read()
        
    app_js = re.sub(r'function parseHousingForm\\(form\\) \\{.*?\\};\\n\\}', 
                    f'function parseHousingForm(form) {{\\n  const data = new FormData(form);\\n  return {{{js_parse}\\n  }};\\n}}', 
                    app_js, flags=re.DOTALL)
                    
    with open('app/web/app.js', 'w', encoding='utf-8') as f:
        f.write(app_js)

if __name__ == "__main__":
    generate_code()
    print("Files completely rebuilt for 62 fields!")
