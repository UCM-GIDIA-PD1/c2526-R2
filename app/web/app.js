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
    m2: Number(data.get("m2")),
    habitaciones: Number(data.get("habitaciones")),
    banos: Number(data.get("banos")),
    codigo_postal: String(data.get("codigo_postal")),
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
    setResult("venta-result", "venta-result-card", "Procesando...");
    const payload = parseHousingForm(form);
    const data = await postJson("/predict/venta", payload);
    setResult(
      "venta-result",
      "venta-result-card",
      `Precio estimado: ${data.prediction} EUR`
    );
  });
}

function initAlquilerForm() {
  const form = document.getElementById("alquiler-form");
  if (!form) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResult("alquiler-result", "alquiler-result-card", "Procesando...");
    const payload = parseHousingForm(form);
    const data = await postJson("/predict/alquiler", payload);
    setResult(
      "alquiler-result",
      "alquiler-result-card",
      `Alquiler estimado: ${data.prediction} EUR/mes`
    );
  });
}

function initTextoForm() {
  const form = document.getElementById("texto-form");
  if (!form) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResult("texto-result", "texto-result-card", "Procesando...");
    const payload = { texto: String(new FormData(form).get("texto")) };
    const data = await postJson("/predict/texto", payload);
    setResult("texto-result", "texto-result-card", `Categoria: ${data.prediction}`);
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
    setResult("imagen-result", "imagen-result-card", "Procesando...");
    const formData = new FormData(form);
    const response = await fetch("/predict/imagen", { method: "POST", body: formData });
    const data = await response.json();
    setResult("imagen-result", "imagen-result-card", `Categoria: ${data.prediction}`);
  });
}

initPrecioTabs();
initVentaForm();
initAlquilerForm();
initTextoForm();
initDragAndDrop();
initImagenForm();
