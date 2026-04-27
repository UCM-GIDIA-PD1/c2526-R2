/* ═══════════════════════════════════════════
   MAiDay — Application Logic
   ═══════════════════════════════════════════ */

const EUR = new Intl.NumberFormat('es-ES', {
  style: 'currency',
  currency: 'EUR',
  maximumFractionDigits: 0,
});

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

function setResult(resultId, cardId, text) {
  const result = document.getElementById(resultId);
  const card = document.getElementById(cardId);
  if (!result || !card) {
    return;
  }
  result.textContent = text;
  card.classList.add("success");
}

function parseHousingForm(form) {
  const data = new FormData(form);
  return {
    Distrito: String(data.get("Distrito")),
    Superficie: Number(data.get("Superficie")),
    Num_habitaciones: Number(data.get("Num_habitaciones")),
    Banyos: Number(data.get("Banyos")),
    Planta: Number(data.get("Planta")),
    Ventanas: String(data.get("Ventanas")),
    Ascensor: Number(data.get("Ascensor")),
    Terraza: Number(data.get("Terraza")),
    Balcon: Number(data.get("Balcon")),
    Orientacion: String(data.get("Orientacion")),
    Consumo: String(data.get("Consumo")),
    Anuncia: String(data.get("Anuncia")),
    lat: Number(data.get("lat")),
    lon: Number(data.get("lon")),
    dist_min_alimentacion: Number(data.get("dist_min_alimentacion")),
    dist_min_bibliotecas: Number(data.get("dist_min_bibliotecas")),
    cantidad_bibliotecas_cerca: Number(data.get("cantidad_bibliotecas_cerca")),
    dist_min_bomberos: Number(data.get("dist_min_bomberos")),
    cantidad_bomberos_cerca: Number(data.get("cantidad_bomberos_cerca")),
    dist_min_cementerios: Number(data.get("dist_min_cementerios")),
    cantidad_cementerios_cerca: Number(data.get("cantidad_cementerios_cerca")),
    dist_min_centros_dia: Number(data.get("dist_min_centros_dia")),
    cantidad_centros_dia_cerca: Number(data.get("cantidad_centros_dia_cerca")),
    dist_min_centros_educativos: Number(data.get("dist_min_centros_educativos")),
    cantidad_centros_educativos_cerca: Number(data.get("cantidad_centros_educativos_cerca")),
    dist_min_centros_mayores: Number(data.get("dist_min_centros_mayores")),
    cantidad_centros_mayores_cerca: Number(data.get("cantidad_centros_mayores_cerca")),
    dist_min_centros_sociales: Number(data.get("dist_min_centros_sociales")),
    cantidad_centros_sociales_cerca: Number(data.get("cantidad_centros_sociales_cerca")),
    dist_min_comercios: Number(data.get("dist_min_comercios")),
    cantidad_comercios_cerca: Number(data.get("cantidad_comercios_cerca")),
    dist_min_comisarias: Number(data.get("dist_min_comisarias")),
    cantidad_comisarias_cerca: Number(data.get("cantidad_comisarias_cerca")),
    dist_min_hospitales: Number(data.get("dist_min_hospitales")),
    cantidad_hospitales_cerca: Number(data.get("cantidad_hospitales_cerca")),
    dist_min_iglesias: Number(data.get("dist_min_iglesias")),
    cantidad_iglesias_cerca: Number(data.get("cantidad_iglesias_cerca")),
    dist_min_negativos: Number(data.get("dist_min_negativos")),
    cantidad_negativos_cerca: Number(data.get("cantidad_negativos_cerca")),
    dist_min_parques: Number(data.get("dist_min_parques")),
    cantidad_parques_cerca: Number(data.get("cantidad_parques_cerca")),
    dist_min_parques_bomberos: Number(data.get("dist_min_parques_bomberos")),
    cantidad_parques_bomberos_cerca: Number(data.get("cantidad_parques_bomberos_cerca")),
    dist_min_polideportivos: Number(data.get("dist_min_polideportivos")),
    cantidad_polideportivos_cerca: Number(data.get("cantidad_polideportivos_cerca")),
    dist_min_puntos_limpios: Number(data.get("dist_min_puntos_limpios")),
    cantidad_puntos_limpios_cerca: Number(data.get("cantidad_puntos_limpios_cerca")),
    dist_min_servicios_sociales: Number(data.get("dist_min_servicios_sociales")),
    cantidad_servicios_sociales_cerca: Number(data.get("cantidad_servicios_sociales_cerca")),
    dist_min_universidades: Number(data.get("dist_min_universidades")),
    cantidad_universidades_cerca: Number(data.get("cantidad_universidades_cerca")),
    dist_min_paradas: Number(data.get("dist_min_paradas")),
    paradas_cerca: Number(data.get("paradas_cerca")),
    lineas_distintas_paradas_cerca: Number(data.get("lineas_distintas_paradas_cerca")),
    dist_min_estaciones: Number(data.get("dist_min_estaciones")),
    lineas_distintas_estaciones_cerca: Number(data.get("lineas_distintas_estaciones_cerca")),
    anio_construccion: Number(data.get("anio_construccion")),
    poblacion_total: Number(data.get("poblacion_total")),
    pct_extranjeros: Number(data.get("pct_extranjeros")),
    pct_mayores_65: Number(data.get("pct_mayores_65")),
    pct_jovenes_30: Number(data.get("pct_jovenes_30")),
  };
}

function initPrecioTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  if (!tabButtons.length) {
    return;
  }

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");

      document.querySelectorAll(".model-card").forEach((card) => {
        card.classList.add("hidden");
      });
      const targetId = button.dataset.tab;
      const activeCard = document.getElementById(targetId);
      if (activeCard) {
        activeCard.classList.remove("hidden");
      }
    });
  });
}

function initVentaForm() {
  const form = document.getElementById("venta-form");
  if (!form) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResult("venta-result", "venta-result-card", "Procesando valoración...");
    const payload = parseHousingForm(form);
    const data = await postJson("/predict/venta", payload);
    const formatted = typeof data.prediction === 'number'
      ? `Precio estimado: ${EUR.format(data.prediction)}`
      : `Precio estimado: ${data.prediction} EUR`;
    setResult("venta-result", "venta-result-card", formatted);
  });
}

function initAlquilerForm() {
  const form = document.getElementById("alquiler-form");
  if (!form) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResult("alquiler-result", "alquiler-result-card", "Procesando valoración...");
    const payload = parseHousingForm(form);
    const data = await postJson("/predict/alquiler", payload);
    const formatted = typeof data.prediction === 'number'
      ? `Alquiler estimado: ${EUR.format(data.prediction)}/mes`
      : `Alquiler estimado: ${data.prediction} EUR/mes`;
    setResult("alquiler-result", "alquiler-result-card", formatted);
  });
}

function initTextoForm() {
  const form = document.getElementById("texto-form");
  if (!form) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResult("texto-result", "texto-result-card", "Procesando clasificación...");
    const payload = { texto: String(new FormData(form).get("texto")) };
    const data = await postJson("/predict/texto", payload);
    setResult("texto-result", "texto-result-card", `Categoría: ${data.prediction}`);
  });
}

function initDragAndDrop() {
  const dropzone = document.getElementById("dropzone");
  const input = document.getElementById("imagen-file");
  const text = document.getElementById("dropzone-text");

  if (!dropzone || !input || !text) {
    return;
  }

  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    if (!event.dataTransfer || !event.dataTransfer.files.length) {
      return;
    }
    input.files = event.dataTransfer.files;
    text.textContent = `Archivo seleccionado: ${input.files[0].name}`;
  });

  input.addEventListener("change", () => {
    if (input.files.length) {
      text.textContent = `Archivo seleccionado: ${input.files[0].name}`;
    }
  });
}

function initImagenForm() {
  const form = document.getElementById("imagen-form");
  if (!form) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResult("imagen-result", "imagen-result-card", "Procesando clasificación...");
    const formData = new FormData(form);
    const response = await fetch("/predict/imagen", { method: "POST", body: formData });
    const data = await response.json();
    setResult("imagen-result", "imagen-result-card", `Categoría: ${data.prediction}`);
  });
}

initPrecioTabs();
initVentaForm();
initAlquilerForm();
initTextoForm();
initDragAndDrop();
initImagenForm();
