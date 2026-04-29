import joblib
from pathlib import Path

m = joblib.load(Path('src/model_artifacts/venta_model.pkl'))
ct = m.named_steps['preprocessor']
print('Numeric columns:', ct.transformers_[0][2])
print('Categorical columns:', ct.transformers_[1][2])
