async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

function parseHousingForm(form) {
  const data = new FormData(form);
  return {
    m2: Number(data.get("m2")),
    habitaciones: Number(data.get("habitaciones")),
    banos: Number(data.get("banos")),
    codigo_postal: String(data.get("codigo_postal")),
  };
}

document.getElementById("venta-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const result = document.getElementById("venta-result");
  result.textContent = "Procesando...";
  const payload = parseHousingForm(e.currentTarget);
  const data = await postJson("/predict/venta", payload);
  result.textContent = JSON.stringify(data, null, 2);
});

document.getElementById("alquiler-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const result = document.getElementById("alquiler-result");
  result.textContent = "Procesando...";
  const payload = parseHousingForm(e.currentTarget);
  const data = await postJson("/predict/alquiler", payload);
  result.textContent = JSON.stringify(data, null, 2);
});

document.getElementById("texto-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const result = document.getElementById("texto-result");
  result.textContent = "Procesando...";
  const payload = {
    texto: String(new FormData(e.currentTarget).get("texto")),
  };
  const data = await postJson("/predict/texto", payload);
  result.textContent = JSON.stringify(data, null, 2);
});

document.getElementById("imagen-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const result = document.getElementById("imagen-result");
  result.textContent = "Procesando...";
  const formData = new FormData(e.currentTarget);
  const response = await fetch("/predict/imagen", { method: "POST", body: formData });
  const data = await response.json();
  result.textContent = JSON.stringify(data, null, 2);
});
