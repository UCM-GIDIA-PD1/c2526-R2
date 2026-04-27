/* ═══════════════════════════════════════════════════════
   mapas.js — Explorador de Mapas interactivo (Leaflet)
   ═══════════════════════════════════════════════════════ */

// ── Estado global ──────────────────────────────────────
let mapa = null;
let capaActual = null;         // L.geoJSON layer activo
let geojsonActual = null;      // datos GeoJSON crudos
let columnasActuales = [];     // columnas de la rejilla activa
let variableActual = "";       // columna seleccionada
let rejillaActual = "";        // rejilla seleccionada

// Escala de colores (Viridis-like, legible en fondo oscuro)
const COLORES = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e",
                 "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];
const NUM_CLASES = COLORES.length;

const MADRID_CENTER = [40.42, -3.70];
const MADRID_ZOOM = 12;

// ── Inicialización ─────────────────────────────────────

function initMapa() {
  mapa = L.map("map", {
    center: MADRID_CENTER,
    zoom: MADRID_ZOOM,
    zoomControl: false,
  });

  // Tiles oscuros (CartoDB Dark Matter)
  L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 19,
    }
  ).addTo(mapa);

  // Control de zoom abajo-derecha
  L.control.zoom({ position: "bottomright" }).addTo(mapa);
}

// ── Carga de capas (catálogo) ──────────────────────────

async function cargarCapas() {
  const res = await fetch("/api/mapas/capas");
  const data = await res.json();
  const rejillas = data.rejillas;

  const sel = document.getElementById("rejilla-select");
  sel.innerHTML = "";

  rejillas.forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r.id;
    opt.textContent = r.nombre;
    sel.appendChild(opt);
  });

  // Guardar en un mapa global para acceso rápido
  window.__capas = {};
  rejillas.forEach((r) => (window.__capas[r.id] = r));

  // Seleccionar barrios por defecto
  sel.value = rejillas[0]?.id || "";
  actualizarSelectoresVariable(sel.value);

  // Event listener
  sel.addEventListener("change", () => {
    const tipo = sel.value;
    actualizarSelectoresVariable(tipo);
    cargarDatos(tipo);
  });

  // Cargar datos iniciales
  await cargarDatos(sel.value);
}

// ── Actualizar selectores de variable ──────────────────

function actualizarSelectoresVariable(tipoRejilla) {
  const capa = window.__capas[tipoRejilla];
  if (!capa) return;

  columnasActuales = capa.columnas;

  // Poblar categorías
  const catSel = document.getElementById("categoria-select");
  const categorias = [...new Set(columnasActuales.map((c) => c.categoria))];
  catSel.innerHTML = '<option value="">Todas</option>';
  categorias.forEach((cat) => {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat;
    catSel.appendChild(opt);
  });

  // Poblar variables (todas)
  poblarVariables("");

  catSel.onchange = () => poblarVariables(catSel.value);
}

function poblarVariables(categoriaFiltro) {
  const varSel = document.getElementById("variable-select");
  const filtradas = categoriaFiltro
    ? columnasActuales.filter((c) => c.categoria === categoriaFiltro)
    : columnasActuales;

  varSel.innerHTML = "";

  // Agrupar por categoría
  const grupos = {};
  filtradas.forEach((c) => {
    if (!grupos[c.categoria]) grupos[c.categoria] = [];
    grupos[c.categoria].push(c);
  });

  for (const [cat, cols] of Object.entries(grupos)) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = cat;
    cols.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.nombre;
      opt.textContent = c.label;
      optgroup.appendChild(opt);
    });
    varSel.appendChild(optgroup);
  }

  // Seleccionar variable por defecto
  const defaultVar = filtradas.find((c) => c.nombre === "Media_precio_m2_venta");
  if (defaultVar) {
    varSel.value = defaultVar.nombre;
  } else if (filtradas.length > 0) {
    varSel.value = filtradas[0].nombre;
  }

  variableActual = varSel.value;

  // Si ya hay datos cargados, re-renderizar sin fetch
  if (geojsonActual) {
    renderizarCoropleta(variableActual);
  }

  varSel.onchange = () => {
    variableActual = varSel.value;
    renderizarCoropleta(variableActual);
  };
}

// ── Cargar datos GeoJSON ───────────────────────────────

async function cargarDatos(tipoRejilla) {
  rejillaActual = tipoRejilla;
  mostrarLoading(true);

  try {
    const res = await fetch(`/api/mapas/datos?rejilla=${tipoRejilla}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    geojsonActual = await res.json();

    variableActual = document.getElementById("variable-select").value;
    renderizarCoropleta(variableActual);
  } catch (err) {
    console.error("Error cargando datos:", err);
    geojsonActual = null;
  } finally {
    mostrarLoading(false);
  }
}

// ── Renderizar coropleta ───────────────────────────────

function renderizarCoropleta(variable) {
  if (!geojsonActual || !variable) return;

  // Limpiar capa anterior
  if (capaActual) {
    mapa.removeLayer(capaActual);
    capaActual = null;
  }

  // Extraer valores válidos
  const valores = geojsonActual.features
    .map((f) => f.properties[variable])
    .filter((v) => v != null && !isNaN(v) && isFinite(v));

  if (valores.length === 0) {
    actualizarStats(0, null, null, null);
    actualizarLeyenda([], [], variable);
    return;
  }

  // Calcular breaks (quantiles)
  const sorted = [...valores].sort((a, b) => a - b);
  const breaks = [];
  for (let i = 1; i < NUM_CLASES; i++) {
    const idx = Math.floor((i * sorted.length) / NUM_CLASES);
    breaks.push(sorted[Math.min(idx, sorted.length - 1)]);
  }

  // Escala de color
  const scale = chroma.scale(COLORES).domain([sorted[0], sorted[sorted.length - 1]]);

  function getColor(val) {
    if (val == null || isNaN(val)) return "rgba(255,255,255,0.05)";
    return scale(val).hex();
  }

  function estilo(feature) {
    const val = feature.properties[variable];
    return {
      fillColor: getColor(val),
      weight: 1,
      opacity: 0.7,
      color: "rgba(255,255,255,0.2)",
      fillOpacity: 0.75,
    };
  }

  // Obtener nombre/id de la zona
  const capa = window.__capas[rejillaActual];
  const colId = capa ? capa.columna_id : null;

  function onCadaFeature(feature, layer) {
    layer.on({
      mouseover: (e) => resaltarZona(e, feature, variable, colId),
      mouseout: (e) => desresaltarZona(e),
      click: (e) => mapa.fitBounds(e.target.getBounds()),
    });
  }

  capaActual = L.geoJSON(geojsonActual, {
    style: estilo,
    onEachFeature: onCadaFeature,
  }).addTo(mapa);

  // Ajustar vista al contenido
  try {
    mapa.fitBounds(capaActual.getBounds(), { padding: [20, 20] });
  } catch (_) {
    // Si falla el fitBounds, mantener la vista actual
  }

  // Actualizar estadísticas
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const media = valores.reduce((a, b) => a + b, 0) / valores.length;
  actualizarStats(geojsonActual.features.length, min, max, media);

  // Actualizar leyenda
  actualizarLeyenda(breaks, COLORES, variable, sorted[0], sorted[sorted.length - 1]);
}

// ── Interactividad: hover ──────────────────────────────

function resaltarZona(e, feature, variable, colId) {
  const layer = e.target;
  layer.setStyle({
    weight: 2.5,
    color: "#55d5ff",
    fillOpacity: 0.9,
  });
  layer.bringToFront();

  // Tooltip flotante
  const props = feature.properties;
  const nombre = props["NOMBRE"] || props[colId] || "—";
  const valor = props[variable];
  const valorStr = valor != null && !isNaN(valor) ? formatearNumero(valor) : "Sin datos";

  const colInfo = columnasActuales.find((c) => c.nombre === variable);
  const label = colInfo ? colInfo.label : variable;

  const tooltip = document.getElementById("map-tooltip");
  tooltip.innerHTML = `
    <strong>${nombre}</strong><br/>
    <span class="tooltip-label">${label}:</span>
    <span class="tooltip-value">${valorStr}</span>
  `;
  tooltip.style.display = "block";

  // Posicionar junto al cursor
  const rect = document.getElementById("map").getBoundingClientRect();
  tooltip.style.left = e.originalEvent.clientX - rect.left + 14 + "px";
  tooltip.style.top = e.originalEvent.clientY - rect.top - 10 + "px";
}

function desresaltarZona(e) {
  if (capaActual) {
    capaActual.resetStyle(e.target);
  }
  document.getElementById("map-tooltip").style.display = "none";
}

// ── Leyenda ────────────────────────────────────────────

function actualizarLeyenda(breaks, colores, variable, minVal, maxVal) {
  const legend = document.getElementById("map-legend");
  if (!breaks.length || !colores.length) {
    legend.innerHTML = "<p>Sin datos para esta variable</p>";
    return;
  }

  const colInfo = columnasActuales.find((c) => c.nombre === variable);
  const label = colInfo ? colInfo.label : variable;

  let html = `<p class="legend-title">${label}</p>`;
  html += '<div class="legend-gradient">';

  // Barra de gradiente continua
  const gradientStops = colores.map((c, i) => `${c} ${(i / (colores.length - 1)) * 100}%`).join(", ");
  html += `<div class="legend-bar" style="background: linear-gradient(to right, ${gradientStops});"></div>`;

  html += `<div class="legend-labels">`;
  html += `<span>${formatearNumero(minVal)}</span>`;
  html += `<span>${formatearNumero((minVal + maxVal) / 2)}</span>`;
  html += `<span>${formatearNumero(maxVal)}</span>`;
  html += `</div>`;
  html += "</div>";

  legend.innerHTML = html;
}

// ── Stats ──────────────────────────────────────────────

function actualizarStats(zonas, min, max, media) {
  document.getElementById("stat-zonas").textContent = zonas || "—";
  document.getElementById("stat-min").textContent = min != null ? formatearNumero(min) : "—";
  document.getElementById("stat-max").textContent = max != null ? formatearNumero(max) : "—";
  document.getElementById("stat-media").textContent = media != null ? formatearNumero(media) : "—";
}

// ── Loading spinner ────────────────────────────────────

function mostrarLoading(visible) {
  document.getElementById("map-loading").style.display = visible ? "flex" : "none";
}

// ── Utilidades ─────────────────────────────────────────

function formatearNumero(val) {
  if (val == null || isNaN(val)) return "—";
  if (Math.abs(val) >= 1000000) return (val / 1000000).toFixed(2) + " M";
  if (Math.abs(val) >= 1000) return (val / 1000).toFixed(1) + " K";
  if (Number.isInteger(val)) return val.toLocaleString("es-ES");
  return val.toFixed(2);
}

// ── Arranque ───────────────────────────────────────────

initMapa();
cargarCapas();
