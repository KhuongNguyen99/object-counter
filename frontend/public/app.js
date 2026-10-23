(() => {
  const $ = (id) => document.getElementById(id);
  const configuredBase = globalThis.COUNTWELL_CONFIG?.apiBaseUrl ?? "";
  let apiBase = "";
  let apiConfigError = false;
  if (configuredBase !== "") {
    try {
      if (typeof configuredBase !== "string" || !/^https?:\/\//i.test(configuredBase)) throw new Error("Invalid API address");
      const url = new URL(configuredBase);
      if (!/^https?:$/.test(url.protocol) || !url.hostname || url.username || url.password || url.search || url.hash) throw new Error("Invalid API address");
      apiBase = `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
    } catch { apiConfigError = true; }
  }
  const apiUrl = (path) => !apiBase || !path || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(path) ? path : `${apiBase}${path.startsWith("/") ? "" : "/"}${path}`;
  const storageKey = apiBase ? `countwell.sessionId.${encodeURIComponent(apiBase)}` : "countwell.sessionId";
  const confidenceStoragePrefix = "countwell.confidence.";
  const svgNs = "http://www.w3.org/2000/svg";
  const maxFileBytes = 20 * 1024 * 1024;
  const maxPixels = 30_000_000;
  const supportedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
  const readStoredId = () => { try { return localStorage.getItem(storageKey); } catch { return null; } };
  const storeId = (id) => { try { localStorage.setItem(storageKey, id); } catch { /* Counting stays available in this tab. */ } };
  const forgetId = () => { try { localStorage.removeItem(storageKey); } catch { /* The in-memory session still resets. */ } };
  function readStoredConfidence(modelId) {
    if (!modelId) return null;
    try {
      const value = Number(localStorage.getItem(`${confidenceStoragePrefix}${modelId}`));
      return Number.isFinite(value) && value >= 0.01 && value <= 0.95 ? value : null;
    } catch { return null; }
  }
  const savedId = readStoredId();
  const state = { models: [], device: "", refs: [], selected: 0, referenceEditing: true, targets: [], session: null, review: null, busy: false, detecting: null, polling: null, pollFailures: 0, loadingModels: true, restoring: !!savedId };
  const modelMode = (id) => state.models.find((model) => model.id === id)?.reference_mode || (["sam3", "countgd-plusplus"].includes(id) ? "per_image" : "shared");
  const referenceMode = () => state.session?.reference_mode || modelMode(state.session?.model || $("model-select").value);
  const perImageMode = () => referenceMode() === "per_image";
  const activeRefs = () => perImageMode() ? state.targets : state.refs;
  const requiredReferences = () => null;
  const recommendedReferences = () => perImageMode() && (state.session?.model || $("model-select").value) === "countgd-plusplus" ? 10 : null;
  const referenceLimit = () => 32;
  const referenceLimitMessage = () => perImageMode() ? "Use at most 32 boxes on each photo." : "Use at most 32 reference boxes in one count.";

  function message(text, success = false) {
    const box = $("notice");
    box.textContent = everydayError(text);
    box.classList.toggle("success", success);
    box.hidden = !text;
  }

  function node(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content != null) element.textContent = content;
    return element;
  }

  const referenceReview = new ReferenceBoxEditor({
    getTotal: (ref) => perImageMode() ? (ref?.boxes.length || 0) : state.refs.reduce((sum, item) => sum + item.boxes.length, 0),
    getLimit: referenceLimit,
    getRequired: requiredReferences,
    getRecommended: recommendedReferences,
    limitMessage: referenceLimitMessage,
    isBusy: () => state.busy || state.restoring || !!state.session && (state.session.status === "finished" || !canContinueSession() || hasPending()),
    isDetecting: (ref) => state.detecting === ref,
    message,
    onChange: renderReferences,
    onDone: () => {
      state.referenceEditing = false;
      renderReferences();
      $("reference-thumbs").children[state.selected]?.focus();
    },
  });

  function clearFiles(items) {
    for (const item of items) URL.revokeObjectURL(item.url);
    items.length = 0;
  }

  const localPhotos = new Map();
  const pendingPhotos = new Map();
  const photoUrl = (result) => localPhotos.get(result.id)?.url || apiUrl(result.original_url);

  function clearLocalPhotos() {
    for (const item of localPhotos.values()) URL.revokeObjectURL(item.url);
    for (const items of pendingPhotos.values()) for (const item of items) URL.revokeObjectURL(item.url);
    localPhotos.clear();
    pendingPhotos.clear();
  }

  function useSession(next) {
    if (state.session?.id && state.session.id !== next.id) clearLocalPhotos();
    for (const batch of next.batches || []) {
      const items = pendingPhotos.get(batch.id);
      if (!items || batch.status !== "complete" && batch.status !== "failed") continue;
      pendingPhotos.delete(batch.id);
      batch.results?.forEach((result, index) => {
        const item = items[index];
        if (!item) return;
        if (item.width === result.width && item.height === result.height) localPhotos.set(result.id, item);
        else URL.revokeObjectURL(item.url);
      });
      for (const item of items.slice(batch.results?.length || 0)) URL.revokeObjectURL(item.url);
    }
    while (localPhotos.size > 40 || [...localPhotos.values()].reduce((bytes, item) => bytes + (item.file?.size || 0), 0) > 100 * 1024 * 1024) {
      const [id, item] = localPhotos.entries().next().value;
      localPhotos.delete(id);
      URL.revokeObjectURL(item.url);
    }
    state.session = next;
  }

  function statusName(status) {
    return { queued: "Waiting", running: "Counting", complete: "Complete", failed: "Failed" }[status] || status;
  }

  function formatDate(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  }

  function everydayError(text) {
    const value = String(text || "");
    if (/CountGD|YOLO|SAM3|FastSAM|PET-DINO|checkpoint|CUDA|torch|tensor|traceback|inference|GPU|JSON|image_boxes|reference_boxes|server|model/i.test(value)) {
      return "This step could not be completed. Please try again.";
    }
    return value;
  }

  async function request(path, options) {
    if (apiConfigError) throw new Error("This page could not connect to counting. Ask for help with the app setup.");
    const response = await fetch(apiUrl(path), options);
    let data;
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) {
      const error = new Error(typeof data.detail === "string" ? everydayError(data.detail) : "This step could not be completed. Please try again.");
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function renderModels(defaultId) {
    const select = $("model-select");
    select.replaceChildren();
    for (const model of state.models) {
      const option = new Option("Counting", model.id);
      option.disabled = !model.available;
      select.add(option);
    }
    const preferred = state.models.find((model) => model.id === defaultId && model.available);
    const first = state.models.find((model) => model.available);
    if (preferred || first) select.value = (preferred || first).id;
    updateConfidence();
    $("model-help").textContent = "";
  }

  function updateConfidence() {
    const selected = state.models.find((model) => model.id === $("model-select").value);
    const mode = modelMode(selected?.id);
    const confidence = readStoredConfidence(selected?.id) ?? selected?.default_confidence ?? 0.25;
    $("confidence").value = String(confidence);
    $("confidence-value").textContent = Number(confidence).toFixed(2);
    $("reference-panel").hidden = mode === "none";
    if ($("reference-panel").hidden) referenceReview.cancel();
    $("text-prompt-note").hidden = mode !== "none";
    $("object-description-field").hidden = mode === "per_image";
    $("object-label").required = mode === "none";
    $("object-description-optional").hidden = mode === "none";
    renderReferences();
    renderBoxes();
  }

  async function loadModels() {
    try {
      const data = await request("/api/models");
      state.models = data.models || [];
      state.device = data.device || "";
      renderModels(data.default_model);
      if (!state.models.some((model) => model.available)) message("Counting is unavailable right now. Try reloading the page.");
    } catch (error) {
      message(`Could not get ready: ${error.message} Try reloading the page.`);
      $("model-help").textContent = "Counting is unavailable right now.";
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
    return { file: normalized, url: URL.createObjectURL(normalized), width: decoded.width, height: decoded.height, boxes: [], candidates: [] };
  }

  async function suggestBoxes(ref) {
    const areas = (ref.detectionAreas || []).map((box) => [...box]);
    if (!areas.length) throw new Error("Mark an area to search first");
    if (!areas.every((box) => box.length === 4 && box.every(Number.isInteger) &&
        0 <= box[0] && box[0] + 4 <= box[2] && box[2] <= ref.width &&
        0 <= box[1] && box[1] + 4 <= box[3] && box[3] <= ref.height)) {
      throw new Error("Keep the marked areas inside this photo");
    }
    const decoded = await decodeImage(ref.file), proposals = [], warnings = [];
    try {
      if (decoded.width !== ref.width || decoded.height !== ref.height) throw new Error("The selected areas did not match this photo");
      for (const [x1, y1, x2, y2] of areas) {
        const width = x2 - x1, height = y2 - y1, canvas = document.createElement("canvas");
        canvas.width = width; canvas.height = height;
        canvas.getContext("2d").drawImage(decoded.source, x1, y1, width, height, 0, 0, width, height);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.94));
        if (!blob || blob.size > maxFileBytes) throw new Error("The selected area could not be prepared");
        const form = new FormData();
        form.append("image", new File([blob], ref.file.name, { type: "image/jpeg" }));
        const data = await request("/api/reference-boxes", { method: "POST", body: form });
        if (data.width !== width || data.height !== height || !Array.isArray(data.boxes) ||
            !data.boxes.every((box) => Array.isArray(box) && box.length === 4 && box.every((value) => typeof value === "number" && Number.isFinite(value)) &&
              0 <= box[0] && box[0] < box[2] && box[2] <= width && 0 <= box[1] && box[1] < box[3] && box[3] <= height)) {
          throw new Error("The suggestions did not fit the marked area. Please try again");
        }
        proposals.push(...data.boxes.map(([left, top, right, bottom]) => [left + x1, top + y1, right + x1, bottom + y1]));
        if (data.boxes.length && typeof data.warning === "string" && data.warning) warnings.push(data.warning);
      }
    } finally {
      if (decoded.source.close) decoded.source.close();
      if (decoded.fallbackUrl) URL.revokeObjectURL(decoded.fallbackUrl);
    }
    const selected = ref.boxes.map((box) => [...box]), seen = [...selected], candidates = [];
    for (const box of proposals) {
      // Overlapping search areas can propose slightly different boxes for the same object.
      const duplicate = seen.some((other) => {
        const overlap = Math.max(0, Math.min(box[2], other[2]) - Math.max(box[0], other[0])) *
          Math.max(0, Math.min(box[3], other[3]) - Math.max(box[1], other[1]));
        return overlap / ((box[2] - box[0]) * (box[3] - box[1]) +
          (other[2] - other[0]) * (other[3] - other[1]) - overlap) >= 0.85;
      });
      if (!duplicate) { seen.push(box); candidates.push(box); }
    }
    if (recommendedReferences()) {
      const prepared = ReferenceSelection.prepare(candidates, ref.width, ref.height);
      const slots = Math.max(0, recommendedReferences() - selected.length);
      ref.candidates = [...prepared.boxes.slice(slots), ...prepared.candidates];
      ref.boxes = [...selected, ...prepared.boxes.slice(0, slots)];
    } else {
      ref.candidates = candidates;
    }
    referenceReview.reset(ref, true);
    const warning = warnings[0] || "";
    return { proposed: ref.candidates.length + ref.boxes.length, warning };
  }

  async function prepareTarget(file, forPrompts = false) {
    const decoded = await decodeImage(file);
    if (forPrompts) {
      const normalized = await jpegFromDecoded(decoded, file.name);
      return { file: normalized, url: URL.createObjectURL(normalized), width: decoded.width, height: decoded.height,
        boxes: [], candidates: [], normalizedForPrompts: true };
    }
    if (supportedTypes.has(file.type)) {
      if (decoded.source.close) decoded.source.close();
      if (decoded.fallbackUrl) URL.revokeObjectURL(decoded.fallbackUrl);
      return { file, url: URL.createObjectURL(file), width: decoded.width, height: decoded.height };
    }
    const normalized = await jpegFromDecoded(decoded, file.name);
    return { file: normalized, url: URL.createObjectURL(normalized), width: decoded.width, height: decoded.height };
  }

  async function prepareExistingTargetsForPrompts() {
    const issues = [];
    for (let index = 0; index < state.targets.length; index++) {
      const item = state.targets[index];
      try {
        if (!item.normalizedForPrompts) {
          const normalized = await prepareTarget(item.file, true);
          normalized.boxes = item.boxes || [];
          normalized.candidates = item.candidates || [];
          normalized.detectionAreas = item.detectionAreas || [];
          state.targets[index] = normalized;
          URL.revokeObjectURL(item.url);
        }
      } catch (error) { issues.push(`${item.file.name}: ${error.message}`); }
    }
    return issues;
  }

  async function addReferences(files) {
    if (!files.length || state.busy || state.session || perImageMode()) return;
    const remaining = 8 - state.refs.length;
    if (files.length > remaining) message(`Only ${remaining} more example photo${remaining === 1 ? "" : "s"} can be added.`);
    const selected = files.slice(0, remaining);
    if (!selected.length) return;
    state.busy = true;
    render();
    const failed = [];
    for (const file of selected) {
      try {
        const ref = await normalizeReference(file);
        state.refs.push(ref);
        state.selected = state.refs.length - 1;
        state.referenceEditing = true;
        renderReferences();
      } catch (error) { failed.push(`${file.name}: ${error.message}`); }
    }
    state.busy = false;
    if (failed.length) message(`Could not add ${failed.length} example photo${failed.length === 1 ? "" : "s"}. ${failed[0]}`);
    else if (files.length <= remaining) message("");
    render();
    if (state.refs.length) referenceReview.focus();
  }

  async function addTargets(files) {
    if (!files.length || state.busy || state.session?.status === "finished" || hasPending()) return;
    const remaining = 20 - state.targets.length;
    if (files.length > remaining) message(`A batch can have up to 20 photos. ${Math.max(0, remaining)} more can be added.`);
    state.busy = true;
    render();
    const failed = [];
    for (const file of files.slice(0, remaining)) {
      try {
        const item = await prepareTarget(file, perImageMode());
        state.targets.push(item);
        if (perImageMode()) {
          state.selected = state.targets.length - 1;
          state.referenceEditing = true;
        }
      }
      catch (error) { failed.push(`${file.name}: ${error.message}`); }
    }
    state.busy = false;
    if (perImageMode() && state.targets.length) {
      state.selected = Math.max(0, state.targets.findIndex((item) => !item.boxes.length));
      state.referenceEditing = true;
    }
    if (failed.length) message(`Could not add ${failed.length} photo${failed.length === 1 ? "" : "s"}. ${failed[0]}`);
    else if (files.length <= remaining) message("");
    renderTargets();
    render();
    if (perImageMode() && state.targets.length) referenceReview.focus();
  }

  function removeTarget(index) {
    if (state.busy || hasPending()) return;
    referenceReview.cancel();
    const [removed] = state.targets.splice(index, 1);
    if (removed) URL.revokeObjectURL(removed.url);
    state.selected = Math.max(0, Math.min(state.selected, activeRefs().length - 1));
    renderTargets();
    renderReferences();
    renderButtons();
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
      remove.disabled = state.busy || hasPending();
      remove.addEventListener("click", () => {
        removeTarget(index);
      });
      card.append(image, remove);
      queue.append(card);
    });
    $("target-count").textContent = `${state.session ? 0 : state.targets.length} / 20`;
    $("next-count").textContent = `${state.session ? state.targets.length : 0} / 20`;
  }

  function renderBoxes() { referenceReview.render(activeRefs()[state.selected] || null); }

  let referencePanelPlace = "setup";
  function renderReferences() {
    const mode = referenceMode();
    const perImage = mode === "per_image";
    const refs = activeRefs();
    const panelPlace = perImage && state.session ? "session" : "setup";
    if (panelPlace !== referencePanelPlace) {
      $(panelPlace === "session" ? "session-reference-slot" : "setup-main").append($("reference-panel"));
      referencePanelPlace = panelPlace;
    }
    $("reference-panel").hidden = mode === "none" || perImage && (!refs.length || state.session?.status === "finished");
    $("reference-title").textContent = perImage ? "Choose examples in each photo" : "Example photos";
    $("reference-upload-actions").hidden = perImage;
    $("remove-reference").setAttribute("aria-label", perImage ? "Remove photo" : "Remove example photo");
    $("remove-reference").title = perImage ? "Remove photo" : "Remove example photo";
    const list = $("reference-thumbs");
    list.replaceChildren();
    $("reference-count").textContent = `${refs.length} / ${perImage ? 20 : 8}`;
    for (const [index, ref] of refs.entries()) {
      const button = node("button", `reference-thumb${index === state.selected ? " selected" : ""}`);
      button.type = "button";
      button.setAttribute("aria-label", `Edit ${perImage ? "photo" : "reference"} ${index + 1}, ${ref.file.name}, ${ref.boxes.length} boxes`);
      button.setAttribute("aria-pressed", String(index === state.selected));
      const image = node("img");
      image.src = ref.url;
      image.alt = "";
      button.append(image);
      button.addEventListener("click", () => { referenceReview.cancel(); state.selected = index; state.referenceEditing = true; renderReferences(); referenceReview.focus(); });
      list.append(button);
    }
    const ref = refs[state.selected];
    $("reference-editor").hidden = !ref || !state.referenceEditing || perImage && !ref.normalizedForPrompts;
    if (ref) {
      if ($("editor-image").dataset.referenceUrl !== ref.url) {
        $("coordinate-form").reset();
        $("editor-image").dataset.referenceUrl = ref.url;
        $("editor-image").src = ref.url;
      }
      $("editor-filename").textContent = ref.file.name;
      $("coordinate-help").textContent = `${ref.width} × ${ref.height} px · origin at top left`;
      $("box-x1").max = $("box-x2").max = String(ref.width);
      $("box-y1").max = $("box-y2").max = String(ref.height);
    }
    renderBoxes();
    $("reference-gallery").disabled = $("reference-camera").disabled = state.busy || perImage || state.refs.length >= 8;
    for (const label of document.querySelectorAll('label[for="reference-gallery"],label[for="reference-camera"]')) label.classList.toggle("disabled", state.busy || perImage || state.refs.length >= 8);
    $("remove-reference").disabled = state.busy || hasPending();
    if (!ref) $("add-coordinate-box").disabled = $("auto-detect-boxes").disabled = true;
  }

  function hasPending() {
    return !!state.session?.batches?.some((batch) => batch.status === "queued" || batch.status === "running");
  }

  function unreviewedRequired() {
    if (!state.session?.requires_review && state.session?.model !== "crosscheck-countgd-yoloe26x") return 0;
    return (state.session.batches || []).flatMap((batch) => batch.results || []).filter((result) => !result.reviewed).length;
  }

  function canContinueSession() {
    return !state.session || state.loadingModels || state.models.some((model) => model.id === state.session.model && model.available);
  }

  function renderButtons() {
    const locked = state.busy || state.restoring;
    for (const button of document.querySelectorAll(".queue-item button")) button.disabled = locked;
    $("start-button").disabled = locked || state.loadingModels || !state.models.some((model) => model.available);
    $("upload-button").disabled = locked || !state.targets.length || hasPending();
    $("quick-count-button").disabled = $("start-button").disabled;
    $("quick-upload-button").disabled = $("upload-button").disabled;
    $("finish-button").disabled = locked || hasPending() || unreviewedRequired() > 0;
    $("finish-review-hint").hidden = unreviewedRequired() === 0;
    for (const id of ["target-gallery", "target-camera", "next-gallery", "next-camera"]) $(id).disabled = locked || (state.session && (state.session.status === "finished" || !canContinueSession() || hasPending()));
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
    renderReferences();
    renderTargets();
    renderButtons();
  }

  function makeResult(result, countMode = "standard") {
    const card = node("div", "result-card");
    const media = node("div", "result-media");
    const preview = node("button", "result-image");
    preview.type = "button";
    preview.setAttribute("aria-label", `Enlarge result for ${result.filename}`);
    preview.append(ResultAnnotations.render(result, photoUrl(result)), node("span", "preview-hint", "Enlarge"));
    preview.addEventListener("click", () => {
      $("dialog-title").textContent = result.filename;
      $("dialog-visual").replaceChildren(ResultAnnotations.render(result, photoUrl(result)), downloadButton(result));
      $("image-dialog").showModal();
    });
    media.append(preview, downloadButton(result));
    const body = node("div", "result-body");
    const line = node("div", "result-line");
    line.append(node("strong", "", result.filename), node("b", "", result.count === 0 ? "0 objects" : `${result.count} ${result.count === 1 ? "object" : "objects"}`));
    body.append(line);
    const warning = result.warning ? (countMode === "quick" ? everydayError(result.warning) : "Check this photo for missed or overlapping objects.") :
      (result.count === 0 && !result.reviewed ? countMode === "quick" ? "No objects found in the marked areas. Check the photo or try Start counting." : "No objects found. Check this photo and your selected examples." : "");
    if (warning && !result.reviewed) body.append(node("div", "result-warning", warning));
    if (result.id) {
      const actions = node("div", "result-review");
      actions.append(node("span", `review-state${result.reviewed ? " done" : ""}`, result.reviewed ? "Reviewed" : "Not reviewed"));
      const button = node("button", "secondary-button", state.session?.status === "finished" ? "View boxes" : "Review photo");
      button.type = "button";
      button.disabled = state.busy || hasPending();
      button.addEventListener("click", () => openReview(result.id));
      actions.append(button);
      body.append(actions);
    }
    card.append(media, body);
    return card;
  }

  function downloadButton(result) {
    const button = node("button", "image-download", "↓");
    button.type = "button";
    button.title = "Download annotated photo";
    button.setAttribute("aria-label", `Download annotated photo for ${result.filename}`);
    button.addEventListener("click", async () => {
      if (button.disabled) return;
      button.disabled = true;
      try { await ResultAnnotations.download(result, photoUrl(result)); }
      catch (error) { message(`Could not download this photo: ${error.message}`); }
      finally { button.disabled = false; }
    });
    return button;
  }

  function renderSession() {
    const session = state.session;
    $("session-title").textContent = session.status === "finished" ? `${session.label || "Object count"} · final count` : session.label ? `Counting ${session.label}` : "Object count";
    $("session-subtitle").textContent = formatDate(session.created_at);
    $("metric-total").textContent = Number(session.total_count || 0).toLocaleString();
    const failedBatch = (session.batches || []).some((batch) => batch.status === "failed");
    $("metric-status").textContent = session.status === "finished" ? failedBatch ? "Final count · failed batches excluded" : "Final count" : hasPending() ? "Counting these photos" : "Running total";
    const batches = session.batches || [];
    const needsReview = unreviewedRequired() > 0 || batches.some((batch) => batch.status === "failed" || (batch.results || []).some((result) => !result.reviewed && (result.warning || result.count === 0)));
    const latest = batches.at(-1);
    $("metric-latest").textContent = latest?.status === "complete" ? Number(latest.total_count || 0).toLocaleString() : latest ? statusName(latest.status) : "—";
    $("metric-latest-note").textContent = latest ? `Batch ${batches.length}` : "No batch yet";
    $("metric-images").textContent = Number(session.image_count || 0).toLocaleString();
    $("metric-batches").textContent = `${batches.length} ${batches.length === 1 ? "batch" : "batches"}`;
    $("results-state").textContent = hasPending() ? "Counting" : needsReview ? "Review needed" : session.status === "finished" ? "Finished" : "Up to date";
    $("results-state").classList.toggle("error", needsReview);
    if (session.image_count > 0 && session.total_count === 0) $("metric-status").textContent = needsReview ? "No matches · review needed" : "0 objects · reviewed";
    $("export-link").href = apiUrl(`/api/sessions/${encodeURIComponent(session.id)}/export.csv`);
    $("next-batch-panel").hidden = session.status === "finished" || !canContinueSession();
    $("new-count-button").hidden = hasPending() || session.status !== "finished" && canContinueSession();
    const list = $("batch-list");
    list.replaceChildren();
    if (!batches.length) list.append(node("div", "batch-card batch-message", "Add photos to start."));
    batches.forEach((batch, index) => {
      const card = node("section", "batch-card");
      const head = node("div", "batch-head");
      const title = node("div");
      title.append(node("strong", "", `Batch ${index + 1}${batch.count_mode === "quick" ? " · Quick count" : ""}`), node("span", "", `${statusName(batch.status)}${formatDate(batch.created_at) ? " · " + formatDate(batch.created_at) : ""}`));
      const total = node("div", "batch-total");
      total.append(node("strong", batch.status === "complete" ? "" : "processing", batch.status === "complete" ? `${batch.total_count} objects` : statusName(batch.status)));
      head.append(title, total);
      card.append(head);
      if (batch.error) card.append(node("div", "batch-message error", "These photos could not be counted. Add them again to retry. They are not included in the total."));
      if (!batch.results?.length && !batch.error) card.append(node("div", "batch-message", batch.status === "complete" ? "No results." : "Counting…"));
      for (const result of batch.results || []) card.append(makeResult(result, batch.count_mode));
      list.append(card);
    });
  }

  function findResult(imageId) {
    return (state.session?.batches || []).flatMap((batch) => batch.results || []).find((result) => result.id === imageId);
  }

  function reviewFromResult(result) {
    const clampBox = (box) => {
      const bounded = [Math.max(0, Math.min(result.width, box[0])), Math.max(0, Math.min(result.height, box[1])),
                       Math.max(0, Math.min(result.width, box[2])), Math.max(0, Math.min(result.height, box[3]))];
      return bounded[0] < bounded[2] && bounded[1] < bounded[3] ? bounded : null;
    };
    const rawBoxes = (result.detections || []).map((detection) => detection.box);
    const boxes = rawBoxes.map(clampBox).filter(Boolean);
    const clipped = boxes.length !== rawBoxes.length || boxes.some((box, index) => box.some((value, i) => value !== rawBoxes[index][i]));
    const sameBox = (left, right) => left.every((value, index) => value === right[index]);
    const readOnly = state.session?.status === "finished";
    const verification = result.verification;
    const secondaryAvailable = verification?.status === "complete" && Array.isArray(verification.secondary_detections);
    return { imageId: result.id, filename: result.filename, revision: result.review_revision || 0,
      width: result.width, height: result.height, imageUrl: photoUrl(result),
      boxes, savedBoxes: boxes.map((box) => [...box]),
      modelCount: result.model_count ?? result.count, reviewed: !!result.reviewed, clipped,
      secondaryAvailable, secondaryFailed: verification?.status === "failed", showSecondary: false,
      secondaryCount: verification?.secondary_count ?? 0,
      secondaryBoxes: (verification?.secondary_detections || []).map((detection, index) => ({
        index: detection.index ?? index + 1, box: clampBox(detection.box),
      })).filter((item) => item.box),
      candidates: (verification?.secondary_only || []).map((detection) => clampBox(detection.box)).filter(Boolean)
        .map((box) => ({ box, accepted: boxes.some((existing) => sameBox(existing, box)) })),
      history: [], mode: readOnly ? "pan" : "add", scale: 1, readOnly, stale: false, drawing: null, panning: null };
  }

  function reviewMessage(text) { $("review-message").textContent = text; }

  function reviewDirty(review = state.review) {
    return !!review && JSON.stringify(review.boxes) !== JSON.stringify(review.savedBoxes);
  }

  function rememberReview() {
    const review = state.review;
    review.history.push({ boxes: review.boxes.map((box) => [...box]), accepted: review.candidates.map((candidate) => candidate.accepted) });
  }

  function renderReviewOverlay() {
    const review = state.review;
    if (!review) return;
    const overlay = $("review-overlay");
    overlay.replaceChildren();
    overlay.setAttribute("viewBox", `0 0 ${review.width} ${review.height}`);
    overlay.setAttribute("preserveAspectRatio", "none");
    $("review-surface").className = `review-surface ${review.mode}`;
    const rectFor = (box, className) => {
      const rect = document.createElementNS(svgNs, "rect");
      rect.setAttribute("x", box[0]); rect.setAttribute("y", box[1]);
      rect.setAttribute("width", box[2] - box[0]); rect.setAttribute("height", box[3] - box[1]);
      rect.classList.add(className);
      return rect;
    };
    if (review.showSecondary) review.secondaryBoxes.forEach((item) => {
      const rect = rectFor(item.box, "secondary-all");
      rect.setAttribute("aria-label", `Other suggestion ${item.index}, view only`);
      rect.setAttribute("pointer-events", "none");
      overlay.append(rect);
      const label = document.createElementNS(svgNs, "text");
      label.classList.add("secondary-label");
      label.setAttribute("x", Math.max(0, item.box[2] - 36 / review.scale));
      label.setAttribute("y", Math.min(review.height, item.box[1] + 16 / review.scale));
      label.setAttribute("font-size", Math.max(20, 15 / review.scale));
      label.setAttribute("stroke-width", 3 / review.scale);
      label.textContent = `S${item.index}`;
      overlay.append(label);
    });
    review.candidates.forEach((candidate, index) => {
      if (candidate.accepted || review.readOnly) return;
      const rect = rectFor(candidate.box, "suggestion");
      rect.setAttribute("aria-label", `Add suggested object ${index + 1}`);
      rect.addEventListener("pointerdown", (event) => {
        if (review.mode !== "add") return;
        event.stopPropagation();
        if (review.readOnly || review.stale || state.busy) return;
        rememberReview();
        review.boxes.push([...candidate.box]);
        candidate.accepted = true;
        reviewMessage("");
        renderReview();
      });
      overlay.append(rect);
    });
    review.boxes.forEach((box, index) => {
      const rect = rectFor(box, "counted");
      rect.setAttribute("aria-label", `Counted object ${index + 1}`);
      rect.addEventListener("pointerdown", (event) => {
        if (review.mode !== "remove") return;
        event.stopPropagation();
        if (review.readOnly || review.stale || state.busy) return;
        rememberReview();
        review.boxes.splice(index, 1);
        review.candidates.forEach((candidate) => {
          candidate.accepted = review.boxes.some((remaining) => remaining.every((value, i) => value === candidate.box[i]));
        });
        reviewMessage("");
        renderReview();
      });
      overlay.append(rect);
      const label = document.createElementNS(svgNs, "text");
      label.setAttribute("x", Math.max(0, box[0]));
      label.setAttribute("y", Math.min(review.height, box[1] + 16 / review.scale));
      label.setAttribute("font-size", Math.max(20, 15 / review.scale));
      label.setAttribute("stroke-width", 3 / review.scale);
      label.textContent = String(index + 1);
      overlay.append(label);
    });
    if (review.drawing?.box) overlay.append(rectFor(review.drawing.box, "draft"));
  }

  function renderReview() {
    const review = state.review;
    if (!review) return;
    $("review-title").textContent = review.filename;
    $("review-count").textContent = `${review.boxes.length} counted`;
    $("review-secondary-row").hidden = !review.secondaryAvailable && !review.secondaryFailed;
    $("review-show-secondary").disabled = !review.secondaryAvailable;
    $("review-show-secondary").checked = review.showSecondary;
    $("review-secondary-label").textContent = `Show other suggestions${review.secondaryAvailable ? ` (${review.secondaryCount})` : ""}`;
    $("review-legend").textContent = review.secondaryFailed ? "Other suggestions are unavailable. Check the photo." : "Green: counted · Orange: suggestions · Blue: other suggestions";
    $("review-status").textContent = review.readOnly ? "Finished" : reviewDirty(review) ? "Unsaved changes" : review.reviewed ? "Reviewed" : "Not reviewed";
    $("review-zoom").textContent = `${Math.round(review.scale * 100)}%`;
    if ($("review-image").getAttribute("src") !== review.imageUrl) $("review-image").src = review.imageUrl;
    for (const mode of ["add", "remove", "pan"]) {
      const button = $(`review-${mode}`);
      button.classList.toggle("active", review.mode === mode);
      button.setAttribute("aria-pressed", String(review.mode === mode));
      button.disabled = review.stale || state.busy || (review.readOnly && mode !== "pan");
    }
    $("review-undo").disabled = review.readOnly || review.stale || state.busy || !review.history.length;
    $("review-save").disabled = review.readOnly || review.stale || state.busy;
    $("review-confirm").disabled = review.readOnly || review.stale || state.busy;
    $("review-reload").hidden = !review.stale;
    renderReviewOverlay();
  }

  function setReviewScale(scale, initial = false) {
    const review = state.review;
    if (!review) return;
    const viewport = $("review-viewport");
    const oldScale = review.scale;
    const centerX = initial ? review.width / 2 : (viewport.scrollLeft + viewport.clientWidth / 2) / oldScale;
    const centerY = initial ? review.height / 2 : (viewport.scrollTop + viewport.clientHeight / 2) / oldScale;
    review.scale = Math.max(0.04, Math.min(4, scale));
    $("review-surface").style.width = `${review.width * review.scale}px`;
    $("review-surface").style.height = `${review.height * review.scale}px`;
    viewport.scrollLeft = Math.max(0, centerX * review.scale - viewport.clientWidth / 2);
    viewport.scrollTop = Math.max(0, centerY * review.scale - viewport.clientHeight / 2);
    renderReview();
  }

  function openReview(imageId) {
    if (state.busy || hasPending()) return;
    const result = findResult(imageId);
    if (!result) return;
    state.review = reviewFromResult(result);
    const ambiguous = result.verification?.ambiguous_primary_indices?.length || 0;
    reviewMessage(`${state.review.clipped ? "Edge boxes were clipped to this photo. " : ""}${state.review.candidates.length ? "Click orange boxes to add them. " : ""}${ambiguous ? `${ambiguous} crowded overlaps need inspection. ` : ""}Check the whole photo for misses.`);
    $("review-dialog").showModal();
    const viewport = $("review-viewport");
    const fit = Math.min((viewport.clientWidth || 900) / result.width,
                         (viewport.clientHeight || 600) / result.height, 1);
    setReviewScale(fit, true);
  }

  function reviewPoint(event) {
    const review = state.review;
    const rect = $("review-overlay").getBoundingClientRect();
    return [Math.round(Math.max(0, Math.min(review.width, (event.clientX - rect.left) * review.width / rect.width))),
            Math.round(Math.max(0, Math.min(review.height, (event.clientY - rect.top) * review.height / rect.height)))];
  }

  function moveReviewPointer(event) {
    const review = state.review;
    if (review?.panning?.pointerId === event.pointerId) {
      $("review-viewport").scrollLeft = review.panning.left - (event.clientX - review.panning.x);
      $("review-viewport").scrollTop = review.panning.top - (event.clientY - review.panning.y);
      return;
    }
    if (review?.drawing?.pointerId !== event.pointerId) return;
    const point = reviewPoint(event);
    const [x, y] = review.drawing.start;
    review.drawing.box = [Math.min(x, point[0]), Math.min(y, point[1]), Math.max(x, point[0]), Math.max(y, point[1])];
    renderReviewOverlay();
  }

  function finishReviewPointer(event) {
    const review = state.review;
    if (!review) return;
    if (review.panning?.pointerId === event.pointerId) { review.panning = null; return; }
    if (review.drawing?.pointerId !== event.pointerId) return;
    moveReviewPointer(event);
    const box = review.drawing.box;
    review.drawing = null;
    if (box && box[2] - box[0] >= 4 && box[3] - box[1] >= 4) {
      rememberReview();
      review.boxes.push(box);
      reviewMessage("");
    }
    renderReview();
  }

  async function saveReview(reviewed) {
    const review = state.review;
    if (!review || review.readOnly || review.stale || state.busy) return;
    const sessionId = state.session?.id;
    if (!sessionId) return;
    state.busy = true;
    stopPolling();
    renderButtons();
    renderReview();
    try {
      useSession(await request(`/api/sessions/${encodeURIComponent(sessionId)}/images/${encodeURIComponent(review.imageId)}/review`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ revision: review.revision, boxes: review.boxes, reviewed }),
      }));
      const saved = findResult(review.imageId);
      review.revision = saved.review_revision;
      review.reviewed = saved.reviewed;
      review.boxes = saved.detections.map((detection) => [...detection.box]);
      review.savedBoxes = review.boxes.map((box) => [...box]);
      review.history = [];
      reviewMessage(reviewed ? "Review confirmed." : "Draft saved.");
      if (reviewed) { $("review-dialog").close(); state.review = null; }
    } catch (error) {
      if (error.status === 409) {
        try { useSession(await request(`/api/sessions/${encodeURIComponent(sessionId)}`)); }
        catch { /* Keep the unsaved boxes even if refresh fails. */ }
        review.stale = true;
        reviewMessage("This photo changed elsewhere. Your edits are still here; reload the latest version to continue.");
      } else reviewMessage(`Could not save review: ${error.message}`);
    } finally {
      state.busy = false;
      render();
      renderReview();
      schedulePoll();
    }
  }

  async function reloadReview() {
    const review = state.review;
    const sessionId = state.session?.id;
    if (!review || !sessionId || state.busy) return;
    const scale = review.scale;
    state.busy = true;
    renderReview();
    try {
      useSession(await request(`/api/sessions/${encodeURIComponent(sessionId)}`));
      const latest = findResult(review.imageId);
      if (!latest) throw new Error("Photo is no longer in this session");
      state.review = reviewFromResult(latest);
      setReviewScale(scale);
      reviewMessage("");
    } catch (error) {
      reviewMessage(`Could not reload the latest review: ${error.message} Your edits are still here.`);
    } finally {
      state.busy = false;
      render();
      renderReview();
      schedulePoll();
    }
  }

  function closeReview() {
    const review = state.review;
    if (reviewDirty(review)
        && !window.confirm("Discard unsaved review changes?")) return;
    $("review-dialog").close();
    state.review = null;
  }

  function stopPolling() {
    if (state.polling?.timer) clearTimeout(state.polling.timer);
    state.polling?.controller?.abort();
    state.polling = null;
  }

  function schedulePoll() {
    stopPolling();
    if (hasPending() && !state.busy) refreshSession(true);
  }

  async function refreshSession(waitForChange = false) {
    if (!state.session || state.busy) return;
    const id = state.session.id;
    const controller = waitForChange ? new AbortController() : null;
    const pending = controller ? { controller } : null;
    if (pending) state.polling = pending;
    try {
      const path = `/api/sessions/${encodeURIComponent(id)}`;
      const query = waitForChange && state.session.revision ? `?after=${encodeURIComponent(state.session.revision)}&wait=25` : "";
      const updated = await request(path + query, controller ? { signal: controller.signal } : undefined);
      if (state.busy || state.session?.id !== id || pending && state.polling !== pending) return;
      if (pending) state.polling = null;
      useSession(updated);
      state.pollFailures = 0;
      message("");
      render();
      schedulePoll();
    } catch (error) {
      if (pending && state.polling !== pending || error.name === "AbortError" || state.session?.id !== id) return;
      if (pending) state.polling = null;
      if (error.status === 404) {
        clearLocalPhotos();
        forgetId();
        state.session = null;
        message("That counting session is no longer available. Start a new count.");
        render();
      } else {
        message(`Could not refresh results: ${error.message} Retrying automatically.`);
        const delay = Math.min(30_000, 2500 * 2 ** Math.min(state.pollFailures++, 4));
        state.polling = { timer: setTimeout(() => refreshSession(true), delay) };
      }
    }
  }

  async function uploadBatch(countMode = "standard") {
    if (!canContinueSession()) { message("Start a new count to add more photos."); return; }
    if (!state.session || !state.targets.length) return;
    if (hasPending()) { message("Wait for the current batch to finish before adding more photos."); return; }
    if (countMode === "quick" ? !ensureQuickAreas() : perImageMode() && !ensurePerImageSelections()) return;
    const previousBatchCount = state.session.batches?.length || 0;
    const form = new FormData();
    for (const item of state.targets) form.append("images", item.file);
    if (countMode === "quick") {
      form.append("count_mode", "quick");
      form.append("search_areas", JSON.stringify(state.targets.map((item) => item.detectionAreas)));
    } else if (perImageMode()) form.append("image_boxes", JSON.stringify(state.targets.map((item) => item.boxes)));
    state.busy = true;
    stopPolling();
    renderButtons();
    message("");
    try {
      const accepted = await request(`/api/sessions/${encodeURIComponent(state.session.id)}/batches`, { method: "POST", body: form });
      pendingPhotos.set(accepted.id, state.targets.splice(0));
      state.busy = false;
      await refreshSession();
    } catch (error) {
      state.busy = false;
      if (!error.status) {
        try {
          const updated = await request(`/api/sessions/${encodeURIComponent(state.session.id)}`);
          useSession(updated);
          if ((updated.batches?.length || 0) > previousBatchCount) {
            // The new batch may belong to another tab, so its image IDs cannot identify these local files.
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

  function ensurePerImageSelections() {
    const validBoxes = (item) => Array.isArray(item.boxes) && item.boxes.every((box) => Array.isArray(box) && box.length === 4 &&
      box.every(Number.isFinite) && 0 <= box[0] && box[0] < box[2] && box[2] <= item.width &&
      0 <= box[1] && box[1] < box[3] && box[3] <= item.height);
    const invalidBoxes = (item) => !validBoxes(item) || !item.boxes.length || item.boxes.length > referenceLimit() ||
      new Set(item.boxes.map((box) => JSON.stringify(box))).size !== item.boxes.length;
    const missing = state.targets.findIndex((item) => !item.normalizedForPrompts || invalidBoxes(item));
    if (missing < 0) return true;
    state.selected = missing;
    state.referenceEditing = true;
    renderReferences();
    message(!state.targets[missing].normalizedForPrompts
      ? `Photo ${missing + 1} could not be prepared. Remove it and try another file.`
      : state.targets[missing].boxes?.length ? `Photo ${missing + 1} needs 1 to 32 distinct boxes inside the photo.`
      : state.targets[missing].candidates?.length
        ? `Select at least one box on photo ${missing + 1}.`
        : `Draw a box on photo ${missing + 1} before counting.`);
    return false;
  }

  function ensureQuickAreas() {
    const missing = state.targets.findIndex((item) => !item.normalizedForPrompts || !item.detectionAreas?.length);
    if (missing < 0) return true;
    state.selected = missing;
    state.referenceEditing = true;
    renderReferences();
    if (state.targets[missing].normalizedForPrompts) {
      referenceReview.areaMode = true;
      renderBoxes();
      message(`Mark an area in photo ${missing + 1} before using Quick count.`);
      referenceReview.focus();
    } else message(`Photo ${missing + 1} could not be prepared. Remove it and try another file.`);
    return false;
  }

  async function startCount(countMode = "standard") {
    if (state.busy) return;
    referenceReview.cancel();
    renderBoxes();
    const label = $("object-label").value.trim();
    const selected = state.models.find((model) => model.id === $("model-select").value && model.available);
    if (!selected) { message("Counting is unavailable right now. Try reloading the page."); return; }
    if (!label && selected.reference_mode === "none") { message("Describe the object you want to count."); $("object-label").focus(); return; }
    const perImage = modelMode(selected.id) === "per_image";
    const usesReferences = selected.reference_mode !== "none" && !perImage;
    if (countMode !== "quick" && usesReferences && !state.refs.length) { message("Add at least one example photo and draw a box around the object."); return; }
    const refs = countMode === "quick" ? [] : usesReferences ? state.refs : [];
    const missing = refs.findIndex((ref) => !ref.boxes.length);
    if (missing >= 0) { state.selected = missing; state.referenceEditing = true; renderReferences(); message(refs[missing].candidates.length ? "Select the object(s) to use as examples." : `Draw a box on example photo ${missing + 1} before starting.`); return; }
    if (!state.targets.length) { message("Add at least one photo to your first batch."); return; }
    if (countMode === "quick" ? !ensureQuickAreas() : perImage && !ensurePerImageSelections()) return;
    const form = new FormData();
    form.append("label", perImage ? "" : label);
    form.append("model", $("model-select").value);
    form.append("confidence", $("confidence").value);
    form.append("reference_boxes", JSON.stringify(refs.map((ref) => ref.boxes)));
    for (const ref of refs) form.append("references", ref.file);
    state.busy = true;
    render();
    message("");
    try {
      useSession(await request("/api/sessions", { method: "POST", body: form }));
      storeId(state.session.id);
      if (!perImage) clearFiles(state.refs);
      state.busy = false;
      render();
      await uploadBatch(countMode);
    } catch (error) {
      state.busy = false;
      message(`Could not start counting: ${error.message}`);
      render();
    }
  }

  async function finishCount() {
    if (!state.session || state.busy || hasPending()) return;
    if (unreviewedRequired()) { message("Review every photo before finishing this count."); return; }
    if (state.targets.length) { message("Count or remove the selected photos before finishing."); return; }
    state.busy = true;
    stopPolling();
    renderButtons();
    try {
      useSession(await request(`/api/sessions/${encodeURIComponent(state.session.id)}/finish`, { method: "POST" }));
      clearFiles(state.targets);
      message("Count finished.", true);
    } catch (error) {
      try {
        useSession(await request(`/api/sessions/${encodeURIComponent(state.session.id)}`));
      } catch { /* A later retry safely reconciles an interrupted finish request. */ }
      if (state.session.status === "finished") message("Count finished.", true);
      else message(`Could not finish counting: ${error.message} Try again.`);
    }
    state.busy = false;
    render();
  }

  function newCount() {
    if (!state.session || hasPending() || state.session.status !== "finished" && canContinueSession()) return;
    stopPolling();
    clearFiles(state.targets);
    clearFiles(state.refs);
    clearLocalPhotos();
    state.session = null;
    state.selected = 0;
    state.referenceEditing = true;
    forgetId();
    $("object-label").value = "";
    updateConfidence();
    message("");
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function changeModel() {
    if (state.busy || state.session) return;
    referenceReview.cancel();
    state.selected = 0;
    if (!perImageMode()) {
      updateConfidence();
      render();
      return;
    }
    state.targets.forEach((item) => { item.boxes ||= []; item.candidates ||= []; });
    state.busy = true;
    updateConfidence();
    render();
    const issues = await prepareExistingTargetsForPrompts();
    state.busy = false;
    state.selected = Math.max(0, state.targets.findIndex((item) => !item.boxes.length));
    state.referenceEditing = true;
    if (issues.length) message(issues[0]);
    render();
  }

  for (const id of ["reference-gallery", "reference-camera"]) $(id).addEventListener("change", (event) => { const files = [...event.target.files]; event.target.value = ""; addReferences(files); });
  for (const id of ["target-gallery", "target-camera", "next-gallery", "next-camera"]) $(id).addEventListener("change", (event) => { const files = [...event.target.files]; event.target.value = ""; addTargets(files); });
  $("confidence").addEventListener("input", (event) => {
    $("confidence-value").textContent = Number(event.target.value).toFixed(2);
    const modelId = $("model-select").value;
    if (modelId) {
      try { localStorage.setItem(`${confidenceStoragePrefix}${modelId}`, event.target.value); } catch { /* The slider still works when storage is unavailable. */ }
    }
  });
  $("model-select").addEventListener("change", changeModel);
  $("remove-reference").addEventListener("click", () => {
    if (state.busy || hasPending()) return;
    if (perImageMode()) { removeTarget(state.selected); return; }
    referenceReview.cancel();
    const [removed] = state.refs.splice(state.selected, 1);
    if (removed) URL.revokeObjectURL(removed.url);
    state.selected = Math.max(0, Math.min(state.selected, state.refs.length - 1));
    renderReferences();
  });
  $("auto-detect-boxes").addEventListener("click", async () => {
    const ref = activeRefs()[state.selected];
    if (!ref || state.busy || referenceReview.edit || !ref.detectionAreas?.length) return;
    state.busy = true;
    state.detecting = ref;
    render();
    message(`Finding objects in ${ref.file.name}…`, true);
    try {
      const result = await suggestBoxes(ref);
      message(result.warning || (result.proposed ? "" : "No objects found. Draw a box manually."));
    } catch (error) { message(`Could not find objects: ${error.message}. Draw boxes manually or retry.`); }
    finally { state.detecting = null; state.busy = false; render(); }
  });
  $("coordinate-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const ref = activeRefs()[state.selected];
    if (!ref || state.busy || referenceReview.edit) return;
    const box = [$("box-x1").valueAsNumber, $("box-y1").valueAsNumber, $("box-x2").valueAsNumber, $("box-y2").valueAsNumber];
    if (!box.every(Number.isInteger) || !(0 <= box[0] && box[0] < box[2] && box[2] <= ref.width && 0 <= box[1] && box[1] < box[3] && box[3] <= ref.height)) {
      message(`Enter a box inside this ${ref.width} × ${ref.height} pixel photo, with right greater than left and bottom greater than top.`);
      return;
    }
    if ((perImageMode() ? ref.boxes.length : state.refs.reduce((sum, item) => sum + item.boxes.length, 0)) >= referenceLimit()) {
      message(referenceLimitMessage()); return;
    }
    referenceReview.addBox(ref, box);
    $("coordinate-form").reset();
    message("");
    renderReferences();
  });
  $("start-button").addEventListener("click", () => startCount());
  $("quick-count-button").addEventListener("click", () => startCount("quick"));
  $("upload-button").addEventListener("click", () => uploadBatch());
  $("quick-upload-button").addEventListener("click", () => uploadBatch("quick"));
  $("finish-button").addEventListener("click", finishCount);
  $("new-count-button").addEventListener("click", newCount);
  $("close-dialog").addEventListener("click", () => $("image-dialog").close());
  $("image-dialog").addEventListener("click", (event) => { if (event.target === $("image-dialog")) $("image-dialog").close(); });
  $("close-review").addEventListener("click", closeReview);
  $("review-dialog").addEventListener("cancel", (event) => {
    const review = state.review;
    if (reviewDirty(review)
        && !window.confirm("Discard unsaved review changes?")) event.preventDefault();
  });
  $("review-dialog").addEventListener("close", () => { state.review = null; });
  for (const mode of ["add", "remove", "pan"]) $(`review-${mode}`).addEventListener("click", () => {
    if (!state.review || (state.review.readOnly && mode !== "pan") || state.review.stale || state.busy) return;
    state.review.mode = mode;
    renderReview();
  });
  $("review-undo").addEventListener("click", () => {
    const review = state.review;
    if (!review || review.readOnly || review.stale || state.busy || !review.history.length) return;
    const previous = review.history.pop();
    review.boxes = previous.boxes;
    review.candidates.forEach((candidate, index) => { candidate.accepted = previous.accepted[index]; });
    renderReview();
  });
  $("review-zoom-in").addEventListener("click", () => { if (state.review) setReviewScale(state.review.scale * 1.5); });
  $("review-zoom-out").addEventListener("click", () => { if (state.review) setReviewScale(state.review.scale / 1.5); });
  $("review-show-secondary").addEventListener("change", (event) => {
    if (!state.review?.secondaryAvailable) return;
    state.review.showSecondary = event.target.checked;
    renderReview();
  });
  $("review-save").addEventListener("click", () => saveReview(false));
  $("review-confirm").addEventListener("click", () => saveReview(true));
  $("review-reload").addEventListener("click", reloadReview);
  $("review-overlay").addEventListener("pointerdown", (event) => {
    const review = state.review;
    if (!review || event.button !== 0 || (review.readOnly && review.mode !== "pan") || review.stale || state.busy) return;
    if (review.mode === "remove") return;
    event.preventDefault();
    $("review-overlay").setPointerCapture(event.pointerId);
    if (review.mode === "pan") {
      review.panning = { pointerId: event.pointerId, x: event.clientX, y: event.clientY,
        left: $("review-viewport").scrollLeft, top: $("review-viewport").scrollTop };
    } else review.drawing = { pointerId: event.pointerId, start: reviewPoint(event), box: null };
  });
  $("review-overlay").addEventListener("pointermove", moveReviewPointer);
  $("review-overlay").addEventListener("pointerup", finishReviewPointer);
  $("review-overlay").addEventListener("pointercancel", (event) => {
    if (!state.review) return;
    if (state.review.drawing?.pointerId === event.pointerId) state.review.drawing = null;
    if (state.review.panning?.pointerId === event.pointerId) state.review.panning = null;
    renderReviewOverlay();
  });
  render();
  loadModels();
  if (savedId) {
    message("Restoring your previous count…", true);
    request(`/api/sessions/${encodeURIComponent(savedId)}`).then((session) => {
      useSession(session);
      state.restoring = false;
      message("");
      render();
      schedulePoll();
    }).catch((error) => {
      state.restoring = false;
      if (error.status === 404) { forgetId(); message("Your previous count is no longer available. Start a new count."); }
      else message(`Could not open your saved count: ${error.message} Reload the page to retry.`);
      render();
    });
  }
})();
