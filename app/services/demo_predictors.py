from hashlib import sha256


def predict_venta_demo(m2: float, habitaciones: int, banos: int, codigo_postal: str) -> float:
    base = 1200 * m2 + 15000 * habitaciones + 10000 * banos
    factor_cp = (sum(ord(c) for c in codigo_postal) % 20) / 100
    return round(base * (1 + factor_cp), 2)


def predict_alquiler_demo(
    m2: float, habitaciones: int, banos: int, codigo_postal: str
) -> float:
    base = 9 * m2 + 120 * habitaciones + 75 * banos
    factor_cp = (sum(ord(c) for c in codigo_postal) % 15) / 100
    return round(base * (1 + factor_cp), 2)


def predict_texto_demo(texto: str) -> str:
    normalized = texto.lower()
    if any(word in normalized for word in ["particular", "dueño", "propietario"]):
        return "particular"
    if any(word in normalized for word in ["promotora", "obra nueva"]):
        return "promotora"
    return "intermediario"


def predict_imagen_demo(image_bytes: bytes) -> str:
    labels = ["dormitorio", "cocina", "salon", "bano"]
    digest = sha256(image_bytes).hexdigest()
    idx = int(digest[:2], 16) % len(labels)
    return labels[idx]
