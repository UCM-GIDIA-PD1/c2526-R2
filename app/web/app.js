/* ═══════════════════════════════════════════
   MAiDay — Application Logic
   ═══════════════════════════════════════════ */

const EUR = new Intl.NumberFormat('es-ES', {
  style: 'currency',
  currency: 'EUR',
  maximumFractionDigits: 0,
});

/* ── Friendly labels for computed features ── */
const FEATURE_LABELS = {
  dist_min_alimentacion: 'Dist. alimentación',
  dist_min_bibliotecas: 'Dist. bibliotecas',
  cantidad_bibliotecas_cerca: 'Bibliotecas cerca',
  dist_min_bomberos: 'Dist. bomberos',
  cantidad_bomberos_cerca: 'Bomberos cerca',
  dist_min_cementerios: 'Dist. cementerios',
  cantidad_cementerios_cerca: 'Cementerios cerca',
  dist_min_centros_dia: 'Dist. centros de día',
  cantidad_centros_dia_cerca: 'Centros de día cerca',
  dist_min_centros_educativos: 'Dist. centros educativos',
  cantidad_centros_educativos_cerca: 'Centros educ. cerca',
  dist_min_centros_mayores: 'Dist. centros mayores',
  cantidad_centros_mayores_cerca: 'Centros mayores cerca',
  dist_min_centros_sociales: 'Dist. centros sociales',
  cantidad_centros_sociales_cerca: 'Centros sociales cerca',
  dist_min_comercios: 'Dist. comercios',
  cantidad_comercios_cerca: 'Comercios cerca',
  dist_min_comisarias: 'Dist. comisarías',
  cantidad_comisarias_cerca: 'Comisarías cerca',
  dist_min_hospitales: 'Dist. hospitales',
  cantidad_hospitales_cerca: 'Hospitales cerca',
  dist_min_iglesias: 'Dist. iglesias',
  cantidad_iglesias_cerca: 'Iglesias cerca',
  dist_min_negativos: 'Dist. puntos negativos',
  cantidad_negativos_cerca: 'Negativos cerca',
  dist_min_parques: 'Dist. parques',
  cantidad_parques_cerca: 'Parques cerca',
  dist_min_parques_bomberos: 'Dist. parques bomberos',
  cantidad_parques_bomberos_cerca: 'Parques bomberos cerca',
  dist_min_polideportivos: 'Dist. polideportivos',
  cantidad_polideportivos_cerca: 'Polideportivos cerca',
  dist_min_puntos_limpios: 'Dist. puntos limpios',
  cantidad_puntos_limpios_cerca: 'Puntos limpios cerca',
  dist_min_servicios_sociales: 'Dist. serv. sociales',
  cantidad_servicios_sociales_cerca: 'Serv. sociales cerca',
  dist_min_universidades: 'Dist. universidades',
  cantidad_universidades_cerca: 'Universidades cerca',
  dist_min_paradas: 'Dist. paradas bus',
  paradas_cerca: 'Paradas bus cerca',
  lineas_distintas_paradas_cerca: 'Líneas bus distintas',
  dist_min_estaciones: 'Dist. metro',
  lineas_distintas_estaciones_cerca: 'Líneas metro distintas',
  anio_construccion: 'Año construcción',
  poblacion_total: 'Población total',
  pct_extranjeros: '% Extranjeros',
  pct_mayores_65: '% Mayores de 65',
  pct_jovenes_30: '% Jóvenes < 30',
};

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Error desconocido' }));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function setResult(resultId, cardId, text, isError = false) {
  const result = document.getElementById(resultId);
  const card = document.getElementById(cardId);
  if (!result || !card) return;
  result.innerHTML = text;
  card.classList.remove("success", "error");
  card.classList.add(isError ? "error" : "success");
}

function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  const textSpan = btn.querySelector('.btn-text');
  const spinnerSpan = btn.querySelector('.btn-spinner');
  if (loading) {
    btn.disabled = true;
    btn.classList.add('loading');
    if (textSpan) textSpan.textContent = 'Calculando...';
    if (spinnerSpan) spinnerSpan.style.display = 'inline-block';
  } else {
    btn.disabled = false;
    btn.classList.remove('loading');
    if (spinnerSpan) spinnerSpan.style.display = 'none';
  }
}

function showComputedFeatures(gridId, containerId, features) {
  const grid = document.getElementById(gridId);
  const container = document.getElementById(containerId);
  if (!grid || !container) return;

  grid.innerHTML = '';
  const entries = Object.entries(features);
  if (entries.length === 0) {
    container.style.display = 'none';
    return;
  }

  // Group features by category
  const groups = {
    'Servicios cercanos': {},
    'Transporte': {},
    'Demografía': {},
    'Otros': {},
  };

  for (const [key, value] of entries) {
    const label = FEATURE_LABELS[key] || key;
    if (key.startsWith('dist_min_paradas') || key.startsWith('dist_min_estaciones') ||
        key.includes('paradas_cerca') || key.includes('estaciones_cerca') ||
        key.includes('lineas_distintas')) {
      groups['Transporte'][label] = value;
    } else if (key.startsWith('poblacion') || key.startsWith('pct_')) {
      groups['Demografía'][label] = value;
    } else if (key.startsWith('dist_min_') || key.startsWith('cantidad_')) {
      groups['Servicios cercanos'][label] = value;
    } else {
      groups['Otros'][label] = value;
    }
  }

  for (const [groupName, groupData] of Object.entries(groups)) {
    const dataEntries = Object.entries(groupData);
    if (dataEntries.length === 0) continue;

    const groupEl = document.createElement('div');
    groupEl.className = 'computed-group';
    groupEl.innerHTML = `<h4 class="computed-group__title">${groupName}</h4>`;

    const itemsGrid = document.createElement('div');
    itemsGrid.className = 'computed-group__items';

    for (const [label, val] of dataEntries) {
      const isDistance = label.startsWith('Dist.');
      const formatted = typeof val === 'number'
        ? (isDistance ? `${val.toFixed(0)} m` : (val % 1 !== 0 ? val.toFixed(1) : val))
        : val;

      const item = document.createElement('div');
      item.className = 'computed-item';
      item.innerHTML = `
        <span class="computed-item__label">${label}</span>
        <span class="computed-item__value">${formatted}</span>
      `;
      itemsGrid.appendChild(item);
    }
    groupEl.appendChild(itemsGrid);
    grid.appendChild(groupEl);
  }
  container.style.display = 'block';
}

/* ── Parse simplified form ── */
function parseSimpleForm(form, mode) {
  const data = new FormData(form);
  const payload = {
    Direccion: String(data.get("Direccion")),
    Distrito: String(data.get("Distrito")),
    Superficie: Number(data.get("Superficie")),
    Num_habitaciones: Number(data.get("Num_habitaciones")),
    Banyos: Number(data.get("Banyos")),
    Planta: Number(data.get("Planta")),
    Ventanas: String(data.get("Ventanas")),
    Ascensor: form.querySelector('[name="Ascensor_check"]')?.checked ? 1 : 0,
    Terraza: form.querySelector('[name="Terraza_check"]')?.checked ? 1 : 0,
    Balcon: form.querySelector('[name="Balcon_check"]')?.checked ? 1 : 0,
    Orientacion: String(data.get("Orientacion")),
    Consumo: String(data.get("Consumo")),
    Anuncia: String(data.get("Anuncia")),
  };

  if (mode === 'alquiler') {
    payload.Equipamiento = form.querySelector('[name="Equipamiento_check"]')?.checked ? 1 : 0;
  }

  const anio = data.get("anio_construccion");
  if (anio && anio !== '') {
    payload.anio_construccion = Number(anio);
  }

  return payload;
}

function initPrecioTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  if (!tabButtons.length) return;

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");
      document.querySelectorAll(".model-card").forEach((card) => {
        card.classList.add("hidden");
      });
      const targetId = button.dataset.tab;
      const activeCard = document.getElementById(targetId);
      if (activeCard) activeCard.classList.remove("hidden");
    });
  });
}

function initVentaForm() {
  const form = document.getElementById("venta-form");
  if (!form) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setLoading("venta-submit-btn", true);
    setResult("venta-result", "venta-result-card", "Geocodificando dirección y calculando datos del entorno...");
    document.getElementById("venta-computed").style.display = "none";

    try {
      const payload = parseSimpleForm(form, 'venta');
      const data = await postJson("/predict/venta/simple", payload);
      const mainPrice = typeof data.prediction === 'number'
        ? `Precio estimado: ${EUR.format(data.prediction)}`
        : `Precio estimado: ${data.prediction} EUR`;
        
      const m2Price = data.prediction_m2
        ? `<br><small style="font-size: 0.6em; font-weight: normal; opacity: 0.8;">(${EUR.format(data.prediction_m2)} / m²)</small>`
        : '';
        
      setResult("venta-result", "venta-result-card", mainPrice + m2Price);

      if (data.features_computed) {
        showComputedFeatures("venta-computed-grid", "venta-computed", data.features_computed);
      }
    } catch (err) {
      setResult("venta-result", "venta-result-card", `Error: ${err.message}`, true);
    } finally {
      setLoading("venta-submit-btn", false);
      const btn = document.getElementById("venta-submit-btn");
      const textSpan = btn?.querySelector('.btn-text');
      if (textSpan) textSpan.textContent = 'Obtener Valoración de Venta';
    }
  });
}

function initAlquilerForm() {
  const form = document.getElementById("alquiler-form");
  if (!form) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setLoading("alquiler-submit-btn", true);
    setResult("alquiler-result", "alquiler-result-card", "Geocodificando dirección y calculando datos del entorno...");
    document.getElementById("alquiler-computed").style.display = "none";

    try {
      const payload = parseSimpleForm(form, 'alquiler');
      const data = await postJson("/predict/alquiler/simple", payload);
      const mainPrice = typeof data.prediction === 'number'
        ? `Alquiler estimado: ${EUR.format(data.prediction)}/mes`
        : `Alquiler estimado: ${data.prediction} EUR/mes`;
        
      const m2Price = data.prediction_m2
        ? `<br><small style="font-size: 0.6em; font-weight: normal; opacity: 0.8;">(${EUR.format(data.prediction_m2)} / m² / mes)</small>`
        : '';
        
      setResult("alquiler-result", "alquiler-result-card", mainPrice + m2Price);

      if (data.features_computed) {
        showComputedFeatures("alquiler-computed-grid", "alquiler-computed", data.features_computed);
      }
    } catch (err) {
      setResult("alquiler-result", "alquiler-result-card", `Error: ${err.message}`, true);
    } finally {
      setLoading("alquiler-submit-btn", false);
      const btn = document.getElementById("alquiler-submit-btn");
      const textSpan = btn?.querySelector('.btn-text');
      if (textSpan) textSpan.textContent = 'Obtener Valoración de Alquiler';
    }
  });
}

function initTextoForm() {
  const form = document.getElementById("texto-form");
  if (!form) return;

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

  if (!dropzone || !input || !text) return;

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
    if (!event.dataTransfer || !event.dataTransfer.files.length) return;
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
  if (!form) return;

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
