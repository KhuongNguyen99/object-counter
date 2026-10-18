(() => {
  const $ = (id) => document.getElementById(id);
  const storageKey = "countwell.sessionId";
  const svgNs = "http://www.w3.org/2000/svg";
  const maxFileBytes = 20 * 1024 * 1024;
  const maxPixels = 30_000_000;
  const supportedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
  const readStoredId = () => { try { return localStorage.getItem(storageKey); } catch { return null; } };
  const storeId = (id) => { try { localStorage.setItem(storageKey, id); } catch { /* Counting stays available in this tab. */ } };
  const forgetId = () => { try { localStorage.removeItem(storageKey); } catch { /* The in-memory session still resets. */ } };
  const savedId = readStoredId();
  const state = { models: [], device: "", refs: [], selected: 0, targets: [], session: null, busy: false, detecting: null, polling: null, loadingModels: true, restoring: !!savedId };

  function message(text, success = false) {
    const box = $("notice");
    box.textContent = text;
    box.classList.toggle("success", success);
    box.hidden = !text;
  }

  function node(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content != null) element.textContent = content;
    return element;
  }

  function clearFiles(items) {
    for (const item of items) URL.revokeObjectURL(item.url);
    items.length = 0;
  }

  function statusName(status) {
    return { queued: "Queued", running: "Counting", complete: "Complete", failed: "Failed" }[status] || status;
  }

  function formatDate(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  }

  async function request(path, options) {
    const response = await fetch(path, options);
    let data;
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) {
      const error = new Error(typeof data.detail === "string" ? data.detail : `Request failed (${response.status}).`);
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function renderModels(defaultId) {
    const select = $("model-select");
    select.replaceChildren();
    for (const model of state.models) {
      const option = new Option(`${model.name}${model.available ? "" : " — unavailable"}`, model.id);
      option.disabled = !model.available;
      select.add(option);
    }
    const preferred = state.models.find((model) => model.id === defaultId && model.available);
    const first = state.models.find((model) => model.available);
    if (preferred || first) select.value = (preferred || first).id;
    updateConfidence();
    $("model-help").textContent = state.device ? `Available on ${state.device}.` : "Choose an available model.";
  }

  function updateConfidence() {
    const selected = state.models.find((model) => model.id === $("model-select").value);
    const confidence = selected?.default_confidence ?? 0.25;
    $("confidence").value = String(confidence);
    $("confidence-value").textContent = Number(confidence).toFixed(2);
    $("reference-panel").hidden = selected?.reference_mode === "none";
    $("text-prompt-note").hidden = selected?.reference_mode !== "none";
  }

  async function loadModels() {
    try {
      const data = await request("/api/models");
      state.models = data.models || [];
      state.device = data.device || "";
      renderModels(data.default_model);
      if (!state.models.some((model) => model.available)) message("No counting model is available. Check the server setup and reload.");
    } catch (error) {
      message(`Could not load models: ${error.message} Reload the page to retry.`);
      $("model-help").textContent = "Models could not be loaded.";
    } finally {
      state.loadingModels = false;
      render();
    }
  }

  async function decodeImage(file) {
    if (file.size > maxFileBytes) throw new Error("File is larger than 20 MB");
    let source;
    let fallbackUrl;
    try {
      if (typeof createImageBitmap === "function") source = await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch { /* The image element path below supports browsers without this option. */ }
    if (!source) {
      fallbackUrl = URL.createObjectURL(file);
      source = new Image();
      source.src = fallbackUrl;
      try { await source.decode(); } catch { URL.revokeObjectURL(fallbackUrl); throw new Error("Browser could not open this image"); }
    }
    const width = source.width || source.naturalWidth;
    const height = source.height || source.naturalHeight;
    if (!width || !height || width * height > maxPixels) {
      if (source.close) source.close();
      if (fallbackUrl) URL.revokeObjectURL(fallbackUrl);
      throw new Error("Image must be 30 megapixels or smaller");
    }
    return { source, width, height, fallbackUrl };
  }

  async function jpegFromDecoded(decoded, name) {
    const { source, width, height } = decoded;
    try {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      canvas.getContext("2d").drawImage(source, 0, 0);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.94));
      if (!blob) throw new Error("The image could not be prepared.");
      if (blob.size > maxFileBytes) throw new Error("Converted image is larger than 20 MB");
      return new File([blob], (name.replace(/\.[^.]+$/, "") || "photo") + ".jpg", { type: "image/jpeg" });
    } finally {
      if (source.close) source.close();
      if (decoded.fallbackUrl) URL.revokeObjectURL(decoded.fallbackUrl);
    }
  }

  async function normalizeReference(file) {
    const decoded = await decodeImage(file);
    const normalized = await jpegFromDecoded(decoded, file.name);
    return { file: normalized, url: URL.createObjectURL(normalized), width: decoded.width, height: decoded.height, boxes: [] };
  }

  async function suggestBoxes(ref, description) {
    const form = new FormData();
    form.append("image", ref.file);
    form.append("label", description);
    const data = await request("/api/reference-boxes", { method: "POST", body: form });
    if (data.width !== ref.width || data.height !== ref.height || !Array.isArray(data.boxes) ||
        !data.boxes.every((box) => Array.isArray(box) && box.length === 4 && box.every((value) => typeof value === "number" && Number.isFinite(value)) &&
          0 <= box[0] && box[0] < box[2] && box[2] <= ref.width && 0 <= box[1] && box[1] < box[3] && box[3] <= ref.height)) {
      throw new Error("The suggested boxes did not match this photo");
    }
    const remaining = 32 - state.refs.reduce((sum, item) => sum + item.boxes.length, 0);
    const boxes = data.boxes.slice(0, Math.max(0, remaining));
    if (!ref.boxes.length) ref.boxes.push(...boxes);
    const warning = typeof data.warning === "string" ? data.warning : "";
    return { added: boxes.length, warning: data.boxes.length > remaining ? `${warning ? warning + " " : ""}32-box limit reached.` : warning };
  }

  async function prepareTarget(file) {
    const decoded = await decodeImage(file);
    if (supportedTypes.has(file.type)) {
      if (decoded.source.close) decoded.source.close();
      if (decoded.fallbackUrl) URL.revokeObjectURL(decoded.fallbackUrl);
      return { file, url: URL.createObjectURL(file) };
    }
    const normalized = await jpegFromDecoded(decoded, file.name);
    return { file: normalized, url: URL.createObjectURL(normalized) };
  }

  async function addReferences(files) {
    if (!files.length || state.busy || state.session) return;
    const remaining = 8 - state.refs.length;
    if (files.length > remaining) message(`Only ${remaining} more reference photo${remaining === 1 ? "" : "s"} can be added.`);
    const selected = files.slice(0, remaining);
    if (!selected.length) return;
    const description = $("object-label").value.trim();
    state.busy = true;
    render();
    const failed = [];
    const issues = [];
    let needsDescription = false;
    let suggested = 0;
    for (const file of selected) {
      try {
        const ref = await normalizeReference(file);
        state.refs.push(ref);
        state.selected = state.refs.length - 1;
        renderReferences();
        if (!description) { needsDescription = true; continue; }
        if (state.refs.reduce((sum, item) => sum + item.boxes.length, 0) >= 32) {
          issues.push(`${file.name}: 32-box limit reached. Remove a box to add one here.`);
          continue;
        }
        state.detecting = ref;
        message(`Finding objects in ${file.name}…`, true);
        renderReferences();
        try {
          const result = await suggestBoxes(ref, description);
          suggested += result.added;
          if (result.warning || !result.added) issues.push(`${file.name}: ${result.warning || "No objects found. Draw a box manually."}`);
        } catch (error) { issues.push(`${file.name}: ${error.message}. Draw boxes manually or retry.`); }
        finally { state.detecting = null; renderReferences(); }
      } catch (error) { failed.push(`${file.name}: ${error.message}`); }
    }
    state.busy = false;
    if (failed.length) message(`Could not add ${failed.length} reference photo${failed.length === 1 ? "" : "s"}. ${failed[0]}`);
    else if (needsDescription) message("Describe the object, then use Auto-detect.");
    else if (issues.length) message(`${suggested} box${suggested === 1 ? "" : "es"} suggested. ${issues[0]}`);
    else if (suggested) message(`${suggested} box${suggested === 1 ? "" : "es"} suggested. Review or edit them before counting.`, true);
    else if (files.length <= remaining) message("");
    render();
  }

  async function addTargets(files) {
    if (!files.length || state.busy || state.session?.status === "finished") return;
    const remaining = 20 - state.targets.length;
    if (files.length > remaining) message(`A batch can have up to 20 photos. ${Math.max(0, remaining)} more can be added.`);
    state.busy = true;
    render();
    const failed = [];
    for (const file of files.slice(0, remaining)) {
      try { state.targets.push(await prepareTarget(file)); }
      catch (error) { failed.push(`${file.name}: ${error.message}`); }
    }
    state.busy = false;
    if (failed.length) message(`Could not add ${failed.length} photo${failed.length === 1 ? "" : "s"}. ${failed[0]}`);
    else if (files.length <= remaining) message("");
    renderTargets();
    render();
  }

  function renderTargets() {
    const queue = $(state.session ? "next-queue" : "target-queue");
    queue.replaceChildren();
    state.targets.forEach((item, index) => {
      const card = node("div", "queue-item");
      const image = node("img");
      image.src = item.url;
      image.alt = item.file.name;
      const remove = node("button", "", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${item.file.name}`);
      remove.disabled = state.busy;
      remove.addEventListener("click", () => {
        if (state.busy) return;
        URL.revokeObjectURL(item.url);
        state.targets.splice(index, 1);
        renderTargets();
        renderButtons();
      });
      card.append(image, remove);
      queue.append(card);
    });
    $("target-count").textContent = `${state.session ? 0 : state.targets.length} / 20`;
    $("next-count").textContent = `${state.session ? state.targets.length : 0} / 20`;
  }

  function renderBoxes() {
    const ref = state.refs[state.selected];
    const overlay = $("box-overlay");
    overlay.replaceChildren();
    const markers = $("box-markers");
    markers.replaceChildren();
    const boxList = $("box-list");
    boxList.replaceChildren();
    if (!ref) return;
    overlay.setAttribute("viewBox", `0 0 ${ref.width} ${ref.height}`);
    overlay.setAttribute("preserveAspectRatio", "none");
    ref.boxes.forEach((box, index) => {
      const rect = document.createElementNS(svgNs, "rect");
      rect.setAttribute("x", box[0]);
      rect.setAttribute("y", box[1]);
      rect.setAttribute("width", box[2] - box[0]);
      rect.setAttribute("height", box[3] - box[1]);
      overlay.append(rect);
      const marker = node("span", "box-marker", String(index + 1));
      marker.style.left = `${100 * box[0] / ref.width}%`;
      marker.style.top = `${100 * box[1] / ref.height}%`;
      markers.append(marker);
      const remove = node("button", "box-remove", `${index + 1} ×`);
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove box ${index + 1} from reference ${state.selected + 1}`);
      remove.disabled = state.busy;
      remove.addEventListener("click", () => {
        if (state.busy) return;
        ref.boxes.splice(index, 1);
        renderReferences();
      });
      boxList.append(remove);
    });
    $("box-count").textContent = `${ref.boxes.length} box${ref.boxes.length === 1 ? "" : "es"}`;
    $("undo-box").disabled = state.busy || !ref.boxes.length;
    $("add-coordinate-box").disabled = state.busy || state.refs.reduce((sum, item) => sum + item.boxes.length, 0) >= 32;
    $("auto-detect-boxes").disabled = state.busy || !!ref.boxes.length || state.refs.reduce((sum, item) => sum + item.boxes.length, 0) >= 32;
    $("auto-detect-boxes").textContent = state.detecting === ref ? "Detecting…" : "Auto-detect";
  }

  function renderReferences() {
    const list = $("reference-thumbs");
    list.replaceChildren();
    $("reference-count").textContent = `${state.refs.length} / 8`;
    for (const [index, ref] of state.refs.entries()) {
      const button = node("button", `reference-thumb${index === state.selected ? " selected" : ""}`);
      button.type = "button";
      button.setAttribute("aria-label", `Edit reference ${index + 1}, ${ref.file.name}, ${ref.boxes.length} boxes`);
      button.setAttribute("aria-pressed", String(index === state.selected));
      const image = node("img");
      image.src = ref.url;
      image.alt = "";
      button.append(image);
      button.addEventListener("click", () => { state.selected = index; renderReferences(); });
      list.append(button);
    }
    const ref = state.refs[state.selected];
    $("reference-editor").hidden = !ref;
    if (ref) {
      if ($("editor-image").dataset.referenceUrl !== ref.url) {
        $("coordinate-form").reset();
        $("editor-image").dataset.referenceUrl = ref.url;
      }
      $("editor-image").src = ref.url;
      $("editor-filename").textContent = ref.file.name;
      $("coordinate-help").textContent = `${ref.width} × ${ref.height} px · origin at top left`;
      $("box-x1").max = $("box-x2").max = String(ref.width);
      $("box-y1").max = $("box-y2").max = String(ref.height);
      renderBoxes();
    }
    $("reference-gallery").disabled = $("reference-camera").disabled = state.busy || state.refs.length >= 8;
    for (const label of document.querySelectorAll('label[for="reference-gallery"],label[for="reference-camera"]')) label.classList.toggle("disabled", state.busy || state.refs.length >= 8);
    $("remove-reference").disabled = state.busy;
    $("add-coordinate-box").disabled = state.busy || !ref || state.refs.reduce((sum, item) => sum + item.boxes.length, 0) >= 32;
    $("auto-detect-boxes").disabled = state.busy || !ref || !!ref.boxes.length || state.refs.reduce((sum, item) => sum + item.boxes.length, 0) >= 32;
  }

  let drawing = null;
  function pointFromEvent(event, ref) {
    const rect = $("box-overlay").getBoundingClientRect();
    return [Math.round(Math.max(0, Math.min(ref.width, (event.clientX - rect.left) * ref.width / rect.width))), Math.round(Math.max(0, Math.min(ref.height, (event.clientY - rect.top) * ref.height / rect.height)))];
  }

  function drawDraft(event) {
    if (!drawing || event.pointerId !== drawing.pointerId) return;
    const ref = state.refs[state.selected];
    if (!ref) return;
    const point = pointFromEvent(event, ref);
    const box = [Math.min(drawing.start[0], point[0]), Math.min(drawing.start[1], point[1]), Math.max(drawing.start[0], point[0]), Math.max(drawing.start[1], point[1])];
    renderBoxes();
    const rect = document.createElementNS(svgNs, "rect");
    rect.classList.add("draft");
    rect.setAttribute("x", box[0]);
    rect.setAttribute("y", box[1]);
    rect.setAttribute("width", box[2] - box[0]);
    rect.setAttribute("height", box[3] - box[1]);
    $("box-overlay").append(rect);
    drawing.box = box;
  }

  function finishDrawing(event) {
    if (!drawing || event.pointerId !== drawing.pointerId) return;
    if (state.busy) { drawing = null; renderBoxes(); return; }
    drawDraft(event);
    const box = drawing.box;
    drawing = null;
    if (box && box[2] - box[0] >= 4 && box[3] - box[1] >= 4) {
      const totalBoxes = state.refs.reduce((sum, ref) => sum + ref.boxes.length, 0);
      if (totalBoxes >= 32) message("Use at most 32 reference boxes in one count.");
      else { state.refs[state.selected].boxes.push(box); message(""); }
    }
    renderReferences();
  }

  function hasPending() {
    return !!state.session?.batches?.some((batch) => batch.status === "queued" || batch.status === "running");
  }

  function renderButtons() {
    const locked = state.busy || state.restoring;
    for (const button of document.querySelectorAll(".queue-item button")) button.disabled = locked;
    $("start-button").disabled = locked || state.loadingModels || !state.models.some((model) => model.available);
    $("upload-button").disabled = locked || !state.targets.length || hasPending();
    $("finish-button").disabled = locked || hasPending();
    for (const id of ["target-gallery", "target-camera", "next-gallery", "next-camera"]) $(id).disabled = locked || (state.session && (state.session.status === "finished" || hasPending()));
    $("object-label").disabled = locked && !!$("object-label").value.trim();
    $("model-select").disabled = locked;
    $("confidence").disabled = locked;
  }

  function render() {
    const active = !!state.session;
    $("page-intro").hidden = active || state.restoring;
    $("setup-view").hidden = active || state.restoring;
    $("session-view").hidden = !active || state.restoring;
    $("header-status").textContent = state.restoring ? "Restoring count" : !active ? "Ready to count" : state.session.status === "finished" ? "Count finished" : hasPending() ? "Counting photos" : "Session active";
    if (active) renderSession();
    else renderReferences();
    renderTargets();
    renderButtons();
  }

  function makeResult(result) {
    const card = node("div", "result-card");
    const preview = node("button", "result-image");
    preview.type = "button";
    preview.setAttribute("aria-label", `Enlarge result for ${result.filename}`);
    const image = node("img");
    image.src = result.annotated_url || result.original_url;
    image.alt = `Annotated count for ${result.filename}`;
    preview.append(image, node("span", "preview-hint", "Enlarge"));
    preview.addEventListener("click", () => {
      $("dialog-title").textContent = result.filename;
      $("dialog-image").src = result.annotated_url || result.original_url;
      $("image-dialog").showModal();
    });
    const body = node("div", "result-body");
    const line = node("div", "result-line");
    line.append(node("strong", "", result.filename), node("b", "", result.count === 0 ? "0 detected" : `${result.count} ${result.count === 1 ? "object" : "objects"}`));
    body.append(line);
    if (result.processing?.tiles) {
      const tiles = result.processing.tiles;
      body.append(node("div", "result-meta", `${tiles} ${tiles === 1 ? "tile" : "tiles"}`));
    }
    const meta = node("div", "result-meta", `${result.width} × ${result.height} px`);
    if (result.elapsed_ms != null) meta.textContent += ` · ${(result.elapsed_ms / 1000).toFixed(1)} s`;
    if (result.processing?.tile_size) meta.textContent += ` · tile size ${result.processing.tile_size} px`;
    if (result.processing?.duplicates_removed) meta.textContent += ` · ${result.processing.duplicates_removed} duplicates removed`;
    const warning = result.warning || (result.count === 0 ? "No matches detected. Check the reference or use a model trained for this object." : "");
    if (warning) body.append(node("div", "result-warning", warning));
    const details = node("details", "detection-details");
    details.append(node("summary", "", "Details"), meta);
    if (result.detections?.length) {
      const list = node("ul");
      for (const detection of result.detections) {
        const confidence = typeof detection.confidence === "number" ? `${Math.round(detection.confidence * 100)}% confidence` : "confidence unavailable";
        list.append(node("li", "", `Object ${detection.index}: ${confidence}`));
      }
      details.append(list);
    }
    body.append(details);
    card.append(preview, body);
    return card;
  }

  function renderSession() {
    const session = state.session;
    $("session-title").textContent = session.status === "finished" ? `${session.label} · final count` : `Counting ${session.label}`;
    $("session-subtitle").textContent = `${formatDate(session.created_at)} · ${session.model || "Selected model"}`;
    $("metric-total").textContent = Number(session.total_count || 0).toLocaleString();
    const failedBatch = (session.batches || []).some((batch) => batch.status === "failed");
    $("metric-status").textContent = session.status === "finished" ? failedBatch ? "Final count · failed batches excluded" : "Final count" : hasPending() ? "Current batch processing" : "Running total";
    const batches = session.batches || [];
    const needsReview = batches.some((batch) => batch.status === "failed" || (batch.results || []).some((result) => result.warning || result.count === 0));
    const latest = batches.at(-1);
    $("metric-latest").textContent = latest?.status === "complete" ? Number(latest.total_count || 0).toLocaleString() : latest ? statusName(latest.status) : "—";
    $("metric-latest-note").textContent = latest ? `Batch ${batches.length}` : "No batch yet";
    $("metric-images").textContent = Number(session.image_count || 0).toLocaleString();
    $("metric-batches").textContent = `${batches.length} ${batches.length === 1 ? "batch" : "batches"}`;
    $("results-state").textContent = hasPending() ? "Processing" : needsReview ? "Review needed" : session.status === "finished" ? "Finished" : "Up to date";
    $("results-state").classList.toggle("error", needsReview);
    if (session.image_count > 0 && session.total_count === 0) $("metric-status").textContent = "No matches · review needed";
    $("export-link").href = `/api/sessions/${encodeURIComponent(session.id)}/export.csv`;
    $("next-batch-panel").hidden = session.status === "finished";
    $("new-count-button").hidden = session.status !== "finished";
    const list = $("batch-list");
    list.replaceChildren();
    if (!batches.length) list.append(node("div", "batch-card batch-message", "Add photos to start."));
    batches.forEach((batch, index) => {
      const card = node("section", "batch-card");
      const head = node("div", "batch-head");
      const title = node("div");
      title.append(node("strong", "", `Batch ${index + 1}`), node("span", "", `${statusName(batch.status)}${formatDate(batch.created_at) ? " · " + formatDate(batch.created_at) : ""}`));
      const total = node("div", "batch-total");
      total.append(node("strong", batch.status === "complete" ? "" : "processing", batch.status === "complete" ? `${batch.total_count} objects` : statusName(batch.status)));
      head.append(title, total);
      card.append(head);
      if (batch.error) card.append(node("div", "batch-message error", `${batch.error} Re-upload to retry. Excluded from total.`));
      if (!batch.results?.length && !batch.error) card.append(node("div", "batch-message", batch.status === "complete" ? "No results." : "Processing…"));
      for (const result of batch.results || []) card.append(makeResult(result));
      list.append(card);
    });
  }

  function stopPolling() {
    if (state.polling) clearTimeout(state.polling);
    state.polling = null;
  }

  function schedulePoll() {
    stopPolling();
    if (hasPending() && !state.busy) state.polling = setTimeout(refreshSession, 1600);
  }

  async function refreshSession() {
    if (!state.session || state.busy) return;
    const id = state.session.id;
    try {
      const updated = await request(`/api/sessions/${encodeURIComponent(id)}`);
      if (state.busy || state.session?.id !== id) return;
      state.session = updated;
      message("");
      render();
      schedulePoll();
    } catch (error) {
      if (error.status === 404) {
        forgetId();
        state.session = null;
        message("That counting session is no longer available. Start a new count.");
        render();
      } else {
        message(`Could not refresh results: ${error.message} Retrying automatically.`);
        state.polling = setTimeout(refreshSession, 2500);
      }
    }
  }

  async function uploadBatch() {
    if (!state.session || !state.targets.length) return;
    if (hasPending()) { message("Wait for the current batch to finish before adding more photos."); return; }
    const previousBatchCount = state.session.batches?.length || 0;
    const form = new FormData();
    for (const item of state.targets) form.append("images", item.file);
    state.busy = true;
    stopPolling();
    renderButtons();
    message("");
    try {
      await request(`/api/sessions/${encodeURIComponent(state.session.id)}/batches`, { method: "POST", body: form });
      clearFiles(state.targets);
      state.busy = false;
      await refreshSession();
    } catch (error) {
      state.busy = false;
      if (!error.status) {
        try {
          const updated = await request(`/api/sessions/${encodeURIComponent(state.session.id)}`);
          state.session = updated;
          if ((updated.batches?.length || 0) > previousBatchCount) {
            clearFiles(state.targets);
            message("Your batch was received. Counting is in progress.", true);
            render();
            schedulePoll();
            return;
          }
        } catch { /* Keep the photos queued until the connection recovers. */ }
      }
      message(`Could not upload this batch: ${error.message} Your selected photos are still here; try again.`);
      render();
    }
  }

  async function startCount() {
    if (state.busy) return;
    const label = $("object-label").value.trim();
    if (!label) { message("Name the object you want to count."); $("object-label").focus(); return; }
    if (!state.models.some((model) => model.id === $("model-select").value && model.available)) { message("Choose an available model."); return; }
    const usesReferences = state.models.find((model) => model.id === $("model-select").value)?.reference_mode !== "none";
    if (usesReferences && !state.refs.length) { message("Add at least one reference photo and draw a box around the object."); return; }
    const refs = usesReferences ? state.refs : [];
    const missing = refs.findIndex((ref) => !ref.boxes.length);
    if (missing >= 0) { state.selected = missing; renderReferences(); message(`Draw a box on reference photo ${missing + 1} before starting.`); return; }
    if (!state.targets.length) { message("Add at least one photo to your first batch."); return; }
    const form = new FormData();
    form.append("label", label);
    form.append("model", $("model-select").value);
    form.append("confidence", $("confidence").value);
    form.append("reference_boxes", JSON.stringify(refs.map((ref) => ref.boxes)));
    for (const ref of refs) form.append("references", ref.file);
    state.busy = true;
    render();
    message("");
    try {
      state.session = await request("/api/sessions", { method: "POST", body: form });
      storeId(state.session.id);
      clearFiles(state.refs);
      state.busy = false;
      render();
      await uploadBatch();
    } catch (error) {
      state.busy = false;
      message(`Could not start counting: ${error.message}`);
      render();
    }
  }

  async function finishCount() {
    if (!state.session || state.busy || hasPending()) return;
    if (state.targets.length) { message("Count or remove the selected photos before finishing."); return; }
    state.busy = true;
    stopPolling();
    renderButtons();
    try {
      state.session = await request(`/api/sessions/${encodeURIComponent(state.session.id)}/finish`, { method: "POST" });
      clearFiles(state.targets);
      message("Count finished.", true);
    } catch (error) {
      try {
        state.session = await request(`/api/sessions/${encodeURIComponent(state.session.id)}`);
      } catch { /* A later retry safely reconciles an interrupted finish request. */ }
      if (state.session.status === "finished") message("Count finished.", true);
      else message(`Could not finish counting: ${error.message} Try again.`);
    }
    state.busy = false;
    render();
  }

  function newCount() {
    if (!state.session || state.session.status !== "finished") return;
    stopPolling();
    clearFiles(state.targets);
    clearFiles(state.refs);
    state.session = null;
    state.selected = 0;
    forgetId();
    $("object-label").value = "";
    updateConfidence();
    message("");
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  for (const id of ["reference-gallery", "reference-camera"]) $(id).addEventListener("change", (event) => { const files = [...event.target.files]; event.target.value = ""; addReferences(files); });
  for (const id of ["target-gallery", "target-camera", "next-gallery", "next-camera"]) $(id).addEventListener("change", (event) => { const files = [...event.target.files]; event.target.value = ""; addTargets(files); });
  $("confidence").addEventListener("input", (event) => { $("confidence-value").textContent = Number(event.target.value).toFixed(2); });
  $("model-select").addEventListener("change", updateConfidence);
  $("remove-reference").addEventListener("click", () => { if (state.busy) return; const [removed] = state.refs.splice(state.selected, 1); if (removed) URL.revokeObjectURL(removed.url); state.selected = Math.max(0, Math.min(state.selected, state.refs.length - 1)); renderReferences(); });
  $("undo-box").addEventListener("click", () => { if (state.busy) return; state.refs[state.selected]?.boxes.pop(); renderReferences(); });
  $("auto-detect-boxes").addEventListener("click", async () => {
    const ref = state.refs[state.selected];
    if (!ref || state.busy || ref.boxes.length || state.refs.reduce((sum, item) => sum + item.boxes.length, 0) >= 32) return;
    const description = $("object-label").value.trim();
    if (!description) { message("Describe the object, then use Auto-detect."); $("object-label").focus(); return; }
    state.busy = true;
    state.detecting = ref;
    render();
    message(`Finding objects in ${ref.file.name}…`, true);
    try {
      const result = await suggestBoxes(ref, description);
      message(result.warning || (result.added ? `${result.added} box${result.added === 1 ? "" : "es"} suggested. Review or edit them before counting.` : "No objects found. Draw a box manually."), !!result.added && !result.warning);
    } catch (error) { message(`Auto-detect failed: ${error.message}. Draw boxes manually or retry.`); }
    finally { state.detecting = null; state.busy = false; render(); }
  });
  $("coordinate-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const ref = state.refs[state.selected];
    if (!ref || state.busy) return;
    const box = [$("box-x1").valueAsNumber, $("box-y1").valueAsNumber, $("box-x2").valueAsNumber, $("box-y2").valueAsNumber];
    if (!box.every(Number.isInteger) || !(0 <= box[0] && box[0] < box[2] && box[2] <= ref.width && 0 <= box[1] && box[1] < box[3] && box[3] <= ref.height)) {
      message(`Enter a box inside this ${ref.width} × ${ref.height} pixel photo, with right greater than left and bottom greater than top.`);
      return;
    }
    if (state.refs.reduce((sum, item) => sum + item.boxes.length, 0) >= 32) { message("Use at most 32 reference boxes in one count."); return; }
    ref.boxes.push(box);
    $("coordinate-form").reset();
    message("");
    renderReferences();
  });
  $("box-overlay").addEventListener("pointerdown", (event) => { if (state.busy || !state.refs[state.selected] || event.button !== 0) return; event.preventDefault(); const overlay = $("box-overlay"); overlay.setPointerCapture(event.pointerId); drawing = { pointerId: event.pointerId, start: pointFromEvent(event, state.refs[state.selected]), box: null }; });
  $("box-overlay").addEventListener("pointermove", drawDraft);
  $("box-overlay").addEventListener("pointerup", finishDrawing);
  $("box-overlay").addEventListener("pointercancel", () => { drawing = null; renderBoxes(); });
  $("start-button").addEventListener("click", startCount);
  $("upload-button").addEventListener("click", uploadBatch);
  $("finish-button").addEventListener("click", finishCount);
  $("new-count-button").addEventListener("click", newCount);
  $("close-dialog").addEventListener("click", () => $("image-dialog").close());
  $("image-dialog").addEventListener("click", (event) => { if (event.target === $("image-dialog")) $("image-dialog").close(); });
  render();
  loadModels();
  if (savedId) {
    message("Restoring your previous count…", true);
    request(`/api/sessions/${encodeURIComponent(savedId)}`).then((session) => {
      state.session = session;
      state.restoring = false;
      message("");
      render();
      schedulePoll();
    }).catch((error) => {
      state.restoring = false;
      if (error.status === 404) { forgetId(); message("Your previous session is no longer available. Start a new count."); }
      else message(`Could not restore your session: ${error.message} Reload the page to retry.`);
      render();
    });
  }
})();
