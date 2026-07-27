"use strict";

const state = {
  token: sessionStorage.getItem("fwSdgToken") || "",
  reviewer: sessionStorage.getItem("fwSdgReviewer") || "operator",
  category: "terrestrial_fire_points",
  items: [],
  index: 0,
  offset: 0,
  total: 0,
  productionAction: "pilot",
  previewUrl: null,
  annotationsVisible: true,
  worker: null,
  livePreviewRelpath: null,
  livePreviewUrl: null,
};

const QUALITY_CHECKS = {
  terrestrial_fire_points: {
    terrain_and_scale_realistic: "Terrain français et échelle réalistes",
    fire_smoke_semantics_aligned: "Fumée localisée et raccordée au foyer, aucun faux soleil",
    camera_and_annotations_coherent: "Caméra calibrée et annotations cohérentes",
    occlusion_and_distance_plausible: "Distance et occlusion plausibles",
    lighting_and_render_artifacts_acceptable: "Éclairage crédible, aucun artefact bloquant",
  },
  france_cross_view: {
    terrain_and_scale_realistic: "Terrain français et échelle réalistes",
    fire_smoke_semantics_aligned: "Fumée localisée et raccordée au foyer, aucun faux soleil",
    camera_and_annotations_coherent: "Caméra et position du feu cohérentes",
    occlusion_and_distance_plausible: "Distance et occlusion plausibles",
    lighting_and_render_artifacts_acceptable: "Éclairage crédible, aucun artefact bloquant",
    orthophoto_mnt_photo_coherent: "Photo, orthophoto et MNT décrivent le même site",
  },
  response_engagement: {
    terrain_and_scale_realistic: "Terrain français et échelle réalistes",
    fire_smoke_semantics_aligned: "Contexte feu et fumée cohérent",
    camera_and_annotations_coherent: "Caméra et annotations cohérentes",
    occlusion_and_distance_plausible: "Distance et occlusion plausibles",
    lighting_and_render_artifacts_acceptable: "Éclairage crédible, aucun artefact bloquant",
    actor_identity_and_engagement_credible: "Identité et engagement de l'acteur crédibles",
    actor_visual_fidelity_and_materials_acceptable: "Géométrie, proportions et matériaux de l'acteur crédibles",
    box_tight_and_object_visible: "Objet visible et boîte suffisamment serrée",
  },
  france_incident_days: {
    sources_traceable: "Sources traçables",
    accepted_and_rejected_facts_justified: "Faits acceptés et rejetés justifiés",
    contradictions_explicit: "Contradictions explicites",
    fire_zone_overlay_coherent: "Calque de zone de feu cohérent",
  },
};

const elements = Object.fromEntries(
  [
    "accept", "auth-token", "camera-details", "case-canvas",
    "case-counter", "case-details", "categories", "connection-state", "create-training-release", "empty-state",
    "export-guard", "file-details", "login-dialog", "login-error", "login-form", "logs",
    "live-preview", "live-preview-caption", "live-preview-frame", "logout", "next", "overlays",
    "previous", "preview", "production-progress-value", "progress-detail", "progress-metrics",
    "progress-percent", "progress-title", "reject", "review-notes", "quality-checks",
    "review-state", "reviewer-id", "setup-progress", "setup-progress-facts",
    "setup-readiness", "setup-stages", "start-production",
    "toggle-annotations", "truth-details",
  ].map((id) => [id, document.getElementById(id)])
);

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${state.token}`);
  if (options.body) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  if (response.status === 401) {
    showLogin("Jeton refusé ou expiré.");
    throw new Error("unauthorized");
  }
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).error || message; } catch { /* response was not JSON */ }
    throw new Error(message);
  }
  return response;
}

function showLogin(message = "") {
  elements["login-error"].textContent = message;
  if (!elements["login-dialog"].open) elements["login-dialog"].showModal();
}

function setConnected(connected) {
  elements["connection-state"].textContent = connected ? "Connecté" : "Hors connexion";
  elements["connection-state"].classList.toggle("connected", connected);
}

function formatCount(value) {
  return new Intl.NumberFormat("fr-FR").format(value);
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${formatCount(bytes)} o`;
  const units = ["Kio", "Mio", "Gio", "Tio"];
  let scaled = bytes;
  let unit = -1;
  do {
    scaled /= 1024;
    unit += 1;
  } while (scaled >= 1024 && unit < units.length - 1);
  return `${scaled.toFixed(scaled >= 100 ? 0 : 1)} ${units[unit]}`;
}

function formatDuration(value) {
  const seconds = Number(value || 0);
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${Math.floor(seconds / 60)} min ${(seconds % 60).toFixed(0)} s`;
}

function renderSetupReadiness(worker) {
  const preparation = worker?.input_preparation;
  const progress = preparation?.progress;
  const card = elements["setup-readiness"];
  card.classList.remove("ready", "running", "blocked");
  const title = card.querySelector("strong");
  const detail = card.querySelector(":scope > span:not(.setup-progress)");
  elements["setup-progress"].textContent = progress?.message || "";
  elements["setup-progress-facts"].replaceChildren();
  const progressFacts = [
    ["Index", progress?.indexes_total ? `${progress.indexes_completed || 0}/${progress.indexes_total}` : null],
    ["USD", progress?.candidates_indexed],
    [
      "Assets",
      progress?.main_assets_total
        ? `${progress.main_assets_completed || 0}/${progress.main_assets_total} fichiers principaux`
        : (progress?.assets_locked ?? progress?.assets_validated),
    ],
    ["Sites", progress?.sites_total ? `${progress.sites_completed || 0}/${progress.sites_total}` : null],
    ["Contrats", progress?.contracts_total ? `${progress.contracts_completed || 0}/${progress.contracts_total}` : null],
    ["Courant", progress?.current_site || progress?.current_asset || progress?.current_index],
  ];
  for (const [label, value] of progressFacts) {
    if (value === null || value === undefined || value === "") continue;
    const item = document.createElement("li");
    item.textContent = `${label} ${value}`;
    elements["setup-progress-facts"].append(item);
  }
  elements["setup-stages"].replaceChildren();
  const labels = {
    runtime: "Runtime",
    storage: "Stockage",
    asset_lock: "Assets",
    terrain_catalog: "Catalogue",
  };
  for (const [stage, stageStatus] of Object.entries(worker?.setup || {})) {
    const item = document.createElement("li");
    item.className = stageStatus.state || "pending";
    const name = document.createElement("strong");
    name.textContent = labels[stage] || stage;
    const description = document.createElement("span");
    description.textContent = `${stageStatus.state || "pending"} — ${stageStatus.detail || ""}`;
    item.append(name, description);
    elements["setup-stages"].append(item);
  }
  if (!preparation) {
    title.textContent = "Préparation non exécutée";
    detail.textContent = "Le catalogue pilote doit être préparé avant toute production.";
    card.classList.add("blocked");
    return;
  }
  if (progress?.state === "blocked" || preparation.state === "blocked") {
    const missing = preparation.missing_actor_classes || preparation.missing_environment || progress?.missing_environment || [];
    title.textContent = `Setup bloqué — ${progress?.phase || preparation.phase || "entrée"}`;
    detail.textContent = missing.length
      ? `Manquants : ${missing.join(", ")}`
      : (progress?.message || preparation.reason || "Consultez le journal du pod.");
    card.classList.add("blocked");
    return;
  }
  if (["pending", "preparing"].includes(preparation.state)) {
    title.textContent = `Setup en cours — ${progress?.phase || preparation.phase || "initialisation"}`;
    detail.textContent = progress?.message || preparation.reason || "Aucun livrable n'est encore déclaré prêt.";
    card.classList.add("running");
    return;
  }
  title.textContent = "Setup pilote prêt";
  detail.textContent = preparation.site_count
    ? `${formatCount(preparation.site_count)} sites terrain-backed — pilote uniquement`
    : "Catalogue verrouillé — revue visuelle requise";
  card.classList.add("ready");
}

function clearLivePreview() {
  if (state.livePreviewUrl) URL.revokeObjectURL(state.livePreviewUrl);
  state.livePreviewUrl = null;
  state.livePreviewRelpath = null;
  elements["live-preview-frame"].hidden = true;
  elements["live-preview"].removeAttribute("src");
}

async function loadLivePreview(production) {
  const lastCompleted = production?.current_batch?.progress?.last_completed;
  const relative = lastCompleted?.preview_relpath || null;
  if (!relative) {
    clearLivePreview();
    return;
  }
  if (relative === state.livePreviewRelpath) return;
  const response = await request("/v1/production/preview");
  const blob = await response.blob();
  if (state.livePreviewUrl) URL.revokeObjectURL(state.livePreviewUrl);
  state.livePreviewUrl = URL.createObjectURL(blob);
  state.livePreviewRelpath = relative;
  elements["live-preview"].src = state.livePreviewUrl;
  elements["live-preview-caption"].textContent =
    `Dernier aperçu non validé — ${lastCompleted.case_id}`;
  elements["live-preview-frame"].hidden = false;
}

function renderProductionProgress(production) {
  const current = production?.current_batch;
  const live = current?.progress;
  const totals = production?.totals || {};
  const completed = Number(production?.progress?.completed_cases || 0);
  const inBatch = Number(live?.produced || 0);
  const totalCases = Number(totals.cases || 0);
  const observed = completed + inBatch;
  const percent = totalCases > 0 ? Math.min(100, 100 * observed / totalCases) : 0;
  elements["production-progress-value"].style.width = `${percent}%`;
  elements["progress-percent"].textContent = `${percent.toFixed(1)} %`;
  if (current) {
    elements["progress-title"].textContent = `Génération ${current.stage} — ${current.category}`;
    elements["progress-metrics"].textContent =
      `${formatCount(observed)} / ${formatCount(totalCases)} cas écrits · ` +
      `lot ${formatCount(current.batch_index)} / ${formatCount(current.batch_total)} · ` +
      `${formatCount(inBatch)} / ${formatCount(current.case_count)} dans le lot`;
    const active = live?.current || {};
    elements["progress-detail"].textContent = [
      current.event_id ? `feu ${current.event_id}` : null,
      active.case_id ? `cas ${active.case_id}` : null,
      active.progression || null,
      active.time_of_day || null,
      active.distance_band || null,
      active.occlusion || null,
      live?.state || null,
    ].filter(Boolean).join(" · ") || "Initialisation du lot.";
  } else {
    const labels = {
      idle: "Production inactive",
      queued: "Production en file",
      awaiting_pilot_review: "Pilotes produits — revue requise",
      awaiting_full_review: "Production terminée — revue requise",
      interrupted_recoverable: "Production interrompue — reprise possible",
      failed: "Production en échec",
    };
    elements["progress-title"].textContent = labels[production?.state] || `Production ${production?.state || "inactive"}`;
    elements["progress-metrics"].textContent = totalCases
      ? `${formatCount(completed)} / ${formatCount(totalCases)} cas écrits`
      : "Aucun lot lancé.";
    elements["progress-detail"].textContent = production?.error ||
      "Les compteurs proviennent uniquement des fichiers écrits.";
  }
  loadLivePreview(production).catch(() => clearLivePreview());
}

function configureProductionButton(production, deliverables, worker) {
  const button = elements["start-production"];
  const productionState = production?.state || "idle";
  const preparation = worker?.input_preparation;
  const pilotTotal = deliverables.categories.reduce(
    (total, item) => total + item.pilot_target,
    0,
  );
  const bulkTotal = deliverables.categories.reduce(
    (total, item) => total + Math.max(0, item.target - item.pilot_target),
    0,
  );
  button.disabled = false;
  if (
    !preparation
    || !["existing", "prepared"].includes(preparation.state)
  ) {
    state.productionAction = "none";
    button.disabled = true;
    button.textContent = preparation?.state === "blocked"
      ? "Setup incomplet — pilote verrouillé"
      : "Setup en cours — pilote verrouillé";
  } else if (["queued", "running"].includes(productionState)) {
    state.productionAction = "none";
    button.disabled = true;
    button.textContent = production?.stage === "bulk"
      ? `Production des ${formatCount(bulkTotal)} cas…`
      : `Production des ${formatCount(pilotTotal)} pilotes…`;
  } else if (productionState === "awaiting_full_review") {
    state.productionAction = "none";
    button.disabled = true;
    button.textContent = "Production terminée — revue requise";
  } else if (productionState === "awaiting_pilot_review" || deliverables.categories.some((item) => item.pilot_produced > 0)) {
    const completePilotInventory = deliverables.categories.every(
      (item) => item.pilot_produced >= item.pilot_target,
    );
    if (!completePilotInventory) {
      state.productionAction = "pilot";
      button.textContent = `Reprendre les ${formatCount(pilotTotal)} pilotes`;
    } else if (deliverables.pilot_ready && preparation.bulk_allowed !== true) {
      state.productionAction = "none";
      button.disabled = true;
      button.textContent = "Pilote validé — ajoutez les sites pour le bulk";
    } else if (deliverables.pilot_ready) {
      state.productionAction = "bulk";
      button.textContent = `Lancer les ${formatCount(bulkTotal)} cas restants`;
    } else {
      state.productionAction = "none";
      button.disabled = true;
      button.textContent = `Validez les ${formatCount(pilotTotal)} pilotes`;
    }
  } else {
    state.productionAction = "pilot";
    button.textContent = productionState === "failed"
      ? `Reprendre les ${formatCount(pilotTotal)} pilotes`
      : `Lancer les ${formatCount(pilotTotal)} pilotes`;
  }
}

function renderCategories(status, production) {
  elements.categories.replaceChildren();
  status.categories.forEach((category, categoryIndex) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `category-button${category.category === state.category ? " selected" : ""}`;
    const percent = Math.min(100, (category.produced / category.target) * 100);
    const measurements = category.pilot_measurements || {};
    const rejectionRate = measurements.rejection_rate_reviewed;
    const observedMetrics = Number(measurements.observed_case_count || 0) > 0
      ? `
        <span class="metric-line"><span>Temps pilote moyen</span><span>${formatDuration(measurements.elapsed_seconds_mean)}</span></span>
        <span class="metric-line"><span>Pic VRAM observé</span><span>${measurements.vram_peak_bytes_max ? formatBytes(measurements.vram_peak_bytes_max) : "N/A"}</span></span>
        <span class="metric-line"><span>Octets / cas</span><span>${formatBytes(measurements.case_output_bytes_mean)}</span></span>
        <span class="metric-line"><span>Rejet révisé</span><span>${rejectionRate === null ? "en attente" : `${(100 * Number(rejectionRate)).toFixed(1)} %`}</span></span>`
      : "";
    button.innerHTML = `
      <span class="category-title"><span class="category-number">${categoryIndex + 1}</span>${category.label}</span>
      <span class="category-metrics">
        <span class="metric-line"><span>Produits</span><span>${formatCount(category.produced)} / ${formatCount(category.target)}</span></span>
        <span class="metric-line"><span>Révisés</span><span>${formatCount(category.reviewed)} / ${formatCount(category.target)}</span></span>
        <span class="metric-line accepted"><span>Acceptés</span><span>${formatCount(category.accepted)} / ${formatCount(category.target)}</span></span>
        ${observedMetrics}
      </span>
      <span class="progress-track"><span class="progress-value" style="width:${percent}%"></span></span>`;
    button.addEventListener("click", async () => {
      state.category = category.category;
      state.index = 0;
      state.offset = 0;
      renderCategories(status, production);
      await loadCases(0);
    });
    elements.categories.append(button);
  });
  const ready = status.training_ready_for_integrity_audit === true;
  elements["export-guard"].classList.toggle("ready", ready);
  elements["export-guard"].querySelector("span").textContent = ready
    ? "Audit d’intégrité final disponible"
    : "Livraison training verrouillée";
  elements["create-training-release"].disabled = !ready;
  configureProductionButton(production, status, state.worker);
}

async function loadStatus() {
  const response = await request("/v1/console/status");
  const payload = await response.json();
  state.worker = payload.worker;
  renderSetupReadiness(payload.worker);
  renderProductionProgress(payload.production);
  renderCategories(payload.deliverables, payload.production);
  setConnected(true);
  return payload;
}

async function loadCases(offset = state.offset, preferredId = state.items[state.index]?.case_id) {
  const normalizedOffset = Math.max(0, Number(offset) || 0);
  const response = await request(`/v1/cases?category=${encodeURIComponent(state.category)}&offset=${normalizedOffset}&limit=100`);
  const payload = await response.json();
  state.offset = payload.offset;
  state.total = payload.total;
  state.items = payload.items;
  const preferredIndex = state.items.findIndex((item) => item.case_id === preferredId);
  state.index = preferredIndex >= 0 ? preferredIndex : 0;
  await renderCase();
}

function detailRow(term, value) {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value ?? "—";
  elements["case-details"].append(dt, dd);
}

function overlayColor(label) {
  if (label === "smoke_column_base") return "#49b9cb";
  if (label.includes("fire")) return "#e56a2f";
  if (label.includes("negative")) return "#d95757";
  return "#e0c04c";
}

function renderOverlays(overlays) {
  elements.overlays.replaceChildren();
  elements.overlays.hidden = !state.annotationsVisible;
  for (const overlay of overlays || []) {
    const node = document.createElement("span");
    node.dataset.label = overlay.label || overlay.kind;
    node.style.color = overlayColor(node.dataset.label);
    if (overlay.kind === "point") {
      node.className = "overlay-point";
      node.style.left = `${Number(overlay.x_normalized) * 100}%`;
      node.style.top = `${Number(overlay.y_normalized) * 100}%`;
    } else if (overlay.kind === "box") {
      node.className = "overlay-box";
      node.style.left = `${Number(overlay.x_min) * 100}%`;
      node.style.top = `${Number(overlay.y_min) * 100}%`;
      node.style.width = `${(Number(overlay.x_max) - Number(overlay.x_min)) * 100}%`;
      node.style.height = `${(Number(overlay.y_max) - Number(overlay.y_min)) * 100}%`;
    } else {
      continue;
    }
    elements.overlays.append(node);
  }
}

function syncOverlayBounds() {
  if (elements.preview.hidden || !elements.preview.naturalWidth) {
    elements.overlays.style.inset = "0";
    elements.overlays.style.width = "auto";
    elements.overlays.style.height = "auto";
    return;
  }
  const canvas = elements["case-canvas"].getBoundingClientRect();
  const scale = Math.min(
    canvas.width / elements.preview.naturalWidth,
    canvas.height / elements.preview.naturalHeight,
  );
  const width = elements.preview.naturalWidth * scale;
  const height = elements.preview.naturalHeight * scale;
  elements.overlays.style.inset = "auto";
  elements.overlays.style.left = `${(canvas.width - width) / 2}px`;
  elements.overlays.style.top = `${(canvas.height - height) / 2}px`;
  elements.overlays.style.width = `${width}px`;
  elements.overlays.style.height = `${height}px`;
}

async function renderCase() {
  const item = state.items[state.index];
  const hasItem = Boolean(item);
  elements.preview.hidden = !hasItem;
  elements["empty-state"].hidden = hasItem;
  elements.previous.disabled = !hasItem || (state.index === 0 && state.offset === 0);
  elements.next.disabled = !hasItem || (state.offset + state.index + 1 >= state.total);
  elements["toggle-annotations"].disabled = !hasItem;
  elements["review-notes"].disabled = !hasItem;
  elements.accept.disabled = !hasItem;
  elements.reject.disabled = !hasItem;
  elements["case-details"].replaceChildren();
  elements["file-details"].replaceChildren();
  elements["review-notes"].value = "";
  elements["quality-checks"].replaceChildren();
  renderOverlays([]);
  if (!item) {
    elements["case-counter"].textContent = "Aucun cas produit";
    elements["truth-details"].textContent = "—";
    elements["camera-details"].textContent = "—";
    elements["review-state"].textContent = "Non révisé";
    return;
  }

  elements["case-counter"].textContent = `Cas ${formatCount(state.offset + state.index + 1)} / ${formatCount(state.total)}`;
  detailRow("ID du cas", item.case_id);
  detailRow("Catégorie", item.category);
  detailRow("Seed", String(item.seed));
  detailRow("Origine", item.data_origin);
  if (item.performance) {
    detailRow("Temps mesuré", formatDuration(item.performance.elapsed_seconds));
    detailRow(
      "Pic VRAM",
      item.performance.vram_peak_bytes
        ? formatBytes(item.performance.vram_peak_bytes)
        : "N/A — dossier sans rendu GPU",
    );
    detailRow("Octets réels", formatBytes(item.performance.case_output_bytes));
    detailRow("Échantillons VRAM", formatCount(item.performance.vram_sample_count));
  }
  elements["truth-details"].textContent = JSON.stringify(item.truth || {}, null, 2);
  elements["camera-details"].textContent = JSON.stringify(item.camera || {}, null, 2);
  for (const artifact of item.artifacts || []) {
    const li = document.createElement("li");
    li.textContent = `${artifact.kind || "fichier"} · ${formatBytes(artifact.bytes)}\n${artifact.relpath}\nsha256:${artifact.sha256}`;
    elements["file-details"].append(li);
  }
  const review = item.review;
  for (const [key, label] of Object.entries(QUALITY_CHECKS[item.category] || {})) {
    const wrapper = document.createElement("label");
    wrapper.className = "quality-check";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.dataset.qualityCheck = key;
    checkbox.checked = review?.quality_checks?.[key] === true;
    wrapper.append(checkbox, document.createTextNode(label));
    elements["quality-checks"].append(wrapper);
  }
  elements["review-state"].className = `review-state${review ? ` ${review.decision}` : ""}`;
  elements["review-state"].textContent = review
    ? `${review.decision === "accepted" ? "Validé" : "Rejeté"} par ${review.reviewer}`
    : "Non révisé";
  elements["review-notes"].value = review?.notes || "";
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  const response = await request(`/v1/cases/${encodeURIComponent(item.category)}/${encodeURIComponent(item.case_id)}/preview`);
  state.previewUrl = URL.createObjectURL(await response.blob());
  elements.preview.src = state.previewUrl;
  try { await elements.preview.decode(); } catch { /* browser may already have decoded it */ }
  syncOverlayBounds();
  renderOverlays(item.overlays);
}

async function review(decision) {
  const item = state.items[state.index];
  if (!item) return;
  const qualityChecks = Object.fromEntries(
    [...elements["quality-checks"].querySelectorAll("input[data-quality-check]")]
      .map((input) => [input.dataset.qualityCheck, input.checked]),
  );
  await request(`/v1/cases/${encodeURIComponent(item.category)}/${encodeURIComponent(item.case_id)}/review`, {
    method: "POST",
    body: JSON.stringify({
      decision,
      reviewer: state.reviewer,
      notes: elements["review-notes"].value,
      quality_checks: qualityChecks,
    }),
  });
  await Promise.all([loadCases(), loadStatus()]);
}

async function loadLogs() {
  const response = await request("/v1/logs?tail=200");
  const payload = await response.json();
  elements.logs.textContent = payload.events.length
    ? payload.events.map((event) => `${event.at}  ${event.event}  ${JSON.stringify(event)}`).join("\n")
    : "Aucun événement enregistré.";
  elements.logs.scrollTop = elements.logs.scrollHeight;
}

async function connect() {
  try {
    await loadStatus();
    await Promise.all([loadCases(), loadLogs()]);
    elements["login-dialog"].close();
  } catch (error) {
    setConnected(false);
    if (error.message !== "unauthorized") showLogin(error.message);
  }
}

elements["login-form"].addEventListener("submit", async (event) => {
  event.preventDefault();
  state.token = elements["auth-token"].value.trim();
  state.reviewer = elements["reviewer-id"].value.trim();
  sessionStorage.setItem("fwSdgToken", state.token);
  sessionStorage.setItem("fwSdgReviewer", state.reviewer);
  await connect();
});
elements.logout.addEventListener("click", () => {
  sessionStorage.removeItem("fwSdgToken");
  state.token = "";
  showLogin();
});
elements.previous.addEventListener("click", async () => {
  if (state.index > 0) {
    state.index -= 1;
    await renderCase();
  } else if (state.offset > 0) {
    await loadCases(Math.max(0, state.offset - 100), null);
    state.index = Math.max(0, state.items.length - 1);
    await renderCase();
  }
});
elements.next.addEventListener("click", async () => {
  if (state.index < state.items.length - 1) {
    state.index += 1;
    await renderCase();
  } else if (state.offset + state.items.length < state.total) {
    await loadCases(state.offset + state.items.length, null);
  }
});
elements["toggle-annotations"].addEventListener("click", () => {
  state.annotationsVisible = !state.annotationsVisible;
  elements["toggle-annotations"].setAttribute("aria-pressed", String(state.annotationsVisible));
  elements.overlays.hidden = !state.annotationsVisible;
});
elements.accept.addEventListener("click", () => review("accepted").catch((error) => alert(error.message)));
elements.reject.addEventListener("click", () => review("rejected").catch((error) => alert(error.message)));
elements["start-production"].addEventListener("click", async () => {
  if (state.productionAction === "none") return;
  elements["start-production"].disabled = true;
  try {
    const endpoint = state.productionAction === "bulk" ? "/v1/production/bulk" : "/v1/production/pilot";
    await request(endpoint, { method: "POST" });
    await Promise.all([loadStatus(), loadLogs()]);
  } catch (error) {
    alert(error.message);
  } finally {
    elements["start-production"].disabled = false;
  }
});
elements["create-training-release"].addEventListener("click", async () => {
  elements["create-training-release"].disabled = true;
  try {
    const response = await request("/v1/training/release", { method: "POST" });
    const release = await response.json();
    alert(`Livraison locale ${release.release_id} créée après audit de ${formatCount(release.total_cases)} cas. Aucun transfert effectué.`);
    await loadLogs();
  } catch (error) {
    alert(error.message);
  } finally {
    await loadStatus();
  }
});
window.addEventListener("resize", syncOverlayBounds);

if (state.token) {
  elements["auth-token"].value = state.token;
  elements["reviewer-id"].value = state.reviewer;
  connect();
} else {
  showLogin();
}

setInterval(() => {
  if (!state.token) return;
  loadStatus().catch(() => setConnected(false));
  loadLogs().catch(() => setConnected(false));
}, 2500);
setInterval(() => {
  if (state.token) loadCases().catch(() => setConnected(false));
}, 7000);
