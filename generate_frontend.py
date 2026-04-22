import pandas as pd
from src.utils.funciones_minio import crear_cliente_minio, bajar_minio

def generate_code():
    c = crear_cliente_minio()
    df = bajar_minio(c, 'dataset_ml/precios/ventas', 'df_ventas_arboles.parquet')
    X = df.drop(columns=['Precio'])
    
    # 1. Generate schemas.py content
    schemas_code = 'from pydantic import BaseModel\nfrom typing import Any, Optional\n\n'
    fields = []
    
    html_fields_html = ""
    app_js_parse = ""
    
    for col in X.columns:
        dtype = X[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            py_type = 'float'
            html_type = 'number" step="any'
            js_parse = f'Number(data.get("{col}"))'
        else:
            py_type = 'str'
            html_type = 'text'
            js_parse = f'String(data.get("{col}"))'
            
        fields.append(f'    {col}: {py_type}')
        
        # HTML form structure
        label_text = col.replace('_', ' ').capitalize()
        html_fields_html += f'''
        <label>
          {label_text}
          <input name="{col}" type="{html_type}" placeholder="Ej: val" required>
        </label>'''
        
        # JS parse array
        app_js_parse += f'\n    {col}: {js_parse},'
    
    schemas_code += 'class VentaInput(BaseModel):\n' + '\n'.join(fields) + '\n\n'
    schemas_code += 'class AlquilerInput(BaseModel):\n' + '\n'.join(fields) + '\n\n'
    schemas_code += '''class TextoInput(BaseModel):
    texto: str

class PredictionResponse(BaseModel):
    model_name: str
    prediction: float | str | int
'''

    with open('app/schemas.py', 'w') as f:
        f.write(schemas_code)
        
    # 2. Update routes.py to remote map_payload_to_model
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

    # Replacing form grid contents
    import re
    # Venta form
    html = re.sub(r'<form id="venta-form" class="form-grid">.*?</form>', 
                  f'<form id="venta-form" class="form-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">{html_fields_html}\n        <button type="submit" style="grid-column: 1 / -1;">Predecir precio de venta</button>\n      </form>', 
                  html, flags=re.DOTALL)
                  
    # Alquiler form
    html = re.sub(r'<form id="alquiler-form" class="form-grid">.*?</form>', 
                  f'<form id="alquiler-form" class="form-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">{html_fields_html}\n        <button type="submit" style="grid-column: 1 / -1;">Predecir precio de alquiler</button>\n      </form>', 
                  html, flags=re.DOTALL)

    with open('app/web/precios.html', 'w', encoding='utf-8') as f:
        f.write(html)
        
    # 4. Update JS
    with open('app/web/app.js', 'r', encoding='utf-8') as f:
        app_js = f.read()
        
    app_js = re.sub(r'function parseHousingForm\(form\) \{.*?\}', 
                    f'function parseHousingForm(form) {{\n  const data = new FormData(form);\n  return {{{app_js_parse}\n  }};\n}}', 
                    app_js, flags=re.DOTALL)
                    
    with open('app/web/app.js', 'w', encoding='utf-8') as f:
        f.write(app_js)

if __name__ == "__main__":
    generate_code()
    print("Files completely rebuilt for 62 fields!")
