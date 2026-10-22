/* Boxes use source-image coordinates. Inspector navigation contains only unselected proposals. */
globalThis.ReferenceBoxEditor = class ReferenceBoxEditor {
  constructor(options = {}) {
    this.options = { isBusy: () => false, isDetecting: () => false, message: () => {}, onChange: () => {}, onDone: () => {}, ...options };
    this.drafts = new WeakMap();
    this.ref = null;
    this.edit = null;
    this.drawing = null;
    this.areaMode = false;
    this.zoom = 1;
    this.magnifierZoom = 1;
    this.fitScale = 1;
    this.pan = null;
    this.$ = (id) => document.getElementById(id);
    const editor = this.$("reference-editor");
    editor.addEventListener("keydown", (event) => this.keydown(event));
    editor.addEventListener("pointerdown", (event) => {
      if (!this.ref || this.options.isBusy() || event.button > 1 ||
          event.target.closest('button,input,textarea,select,summary,a,[contenteditable]:not([contenteditable="false"])')) return;
      editor.focus({ preventScroll: true });
    }, { capture: true });
    for (const [id, action] of Object.entries({
      "reference-box-prev": "previous", "reference-box-next": "next", "reference-box-toggle": "keep",
      "reference-box-remove": "remove", "reference-box-adjust": "adjust-candidate", "reference-selected-adjust": "adjust",
      "reference-box-save": "save", "reference-box-cancel": "cancel", "undo-box": "undo", "reference-done": "done",
      "reference-select-areas": "areas",
    })) this.$(id).addEventListener("click", () => this.action(action));
    const overlay = this.$("box-overlay"), viewport = this.$("reference-image-viewport");
    overlay.addEventListener("pointerdown", (event) => this.startDrawing(event));
    overlay.addEventListener("pointermove", (event) => this.draw(event));
    overlay.addEventListener("pointerup", (event) => this.finishDrawing(event));
    overlay.addEventListener("pointercancel", () => this.cancelDrawing());
    viewport.addEventListener("pointermove", (event) => {
      if (this.pan && event.pointerId === this.pan.pointerId) {
        viewport.scrollLeft = this.pan.left - event.clientX + this.pan.x;
        viewport.scrollTop = this.pan.top - event.clientY + this.pan.y;
        this.hideMagnifier();
      } else this.magnify(event);
    });
    viewport.addEventListener("pointerleave", () => this.hideMagnifier());
    viewport.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "touch") this.hideMagnifier();
      if (event.button !== 1 || !this.ref || this.options.isBusy()) return;
      event.preventDefault();
      viewport.setPointerCapture(event.pointerId);
      this.pan = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop };
      viewport.classList.add("panning");
      this.hideMagnifier();
    });
    const endPan = () => { this.pan = null; viewport.classList.remove("panning"); };
    viewport.addEventListener("pointerup", endPan);
    viewport.addEventListener("pointercancel", endPan);
    viewport.addEventListener("auxclick", (event) => { if (event.button === 1) event.preventDefault(); });
    viewport.addEventListener("scroll", () => this.hideMagnifier());
    viewport.addEventListener("wheel", (event) => {
      if (!this.ref || this.options.isBusy() || (this.drawing && !event.altKey)) return;
      event.preventDefault();
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientHeight : 1);
      if (event.altKey) this.magnifierZoom = Math.max(0.5, Math.min(8, this.magnifierZoom * Math.exp(-delta * 0.002)));
      else this.setZoom(this.zoom * Math.exp(-delta * 0.002), event.clientX, event.clientY);
      this.magnify(event);
    }, { passive: false });
    this.$("reference-zoom-in").addEventListener("click", () => this.setZoom(this.zoom * 1.5));
    this.$("reference-zoom-out").addEventListener("click", () => this.setZoom(this.zoom / 1.5));
    this.$("reference-zoom-reset").addEventListener("click", () => this.setZoom(1));
    this.$("editor-image").addEventListener("load", () => this.render(this.ref));
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => { this.layoutImage(); this.hideMagnifier(); }).observe(viewport);
      new ResizeObserver(() => this.renderCrop()).observe(this.$("reference-crop"));
    }
    globalThis.addEventListener?.("blur", () => this.hideMagnifier());
  }

  limit(ref = this.ref) { return this.options.getLimit?.(ref) ?? 32; }
  required(ref = this.ref) { return this.options.getRequired?.(ref) ?? null; }
  total(ref = this.ref) { return this.options.getTotal?.(ref) ?? ref?.boxes.length ?? 0; }
  limitMessage() { return this.options.limitMessage?.() || `Use at most ${this.limit()} example boxes.`; }
  data(ref) { if (!this.drafts.has(ref)) this.reset(ref); return this.drafts.get(ref); }
  queue(data = this.data(this.ref)) { return data.items.flatMap((item, index) => item.selected ? [] : [index]); }

  reset(ref, undoable = false) {
    this.cancel();
    const previous = this.drafts.get(ref);
    const history = undoable && previous ? [...previous.history, this.snapshot(previous)] : [];
    const selected = new Set((ref.boxes || []).map((box) => JSON.stringify(box))), seen = new Set();
    const items = [];
    for (const box of [...(ref.candidates || []), ...(ref.boxes || [])]) {
      const key = JSON.stringify(box);
      if (seen.has(key)) continue;
      seen.add(key);
      items.push({ box: [...box], selected: selected.has(key) });
    }
    this.drafts.set(ref, { items, areas: (ref.detectionAreas || []).map((box) => [...box]), active: items.findIndex((item) => !item.selected), selectedActive: null, history });
  }

  normalize(data) {
    const queue = this.queue(data);
    if (!queue.includes(data.active)) data.active = queue.find((index) => index >= data.active) ?? queue[queue.length - 1] ?? -1;
    if (!data.items[data.selectedActive]?.selected) data.selectedActive = null;
    return queue;
  }

  snapshot(data) {
    return { items: data.items.map((item) => ({ box: [...item.box], selected: item.selected })), areas: data.areas.map((box) => [...box]), active: data.active, selectedActive: data.selectedActive };
  }
  remember(data) { data.history.push(this.snapshot(data)); }
  sync(ref) {
    const data = this.data(ref);
    this.normalize(data);
    ref.boxes = data.items.filter((item) => item.selected).map((item) => [...item.box]);
    ref.candidates = data.items.filter((item) => !item.selected).map((item) => [...item.box]);
    ref.detectionAreas = data.areas.map((box) => [...box]);
  }
  announce(text) { this.$("reference-announcement").textContent = text; }
  changed() { this.options.onChange(); this.render(this.ref); }

  cancelDrawing() {
    const drawing = this.drawing;
    this.drawing = null;
    if (drawing && this.$("box-overlay").hasPointerCapture?.(drawing.pointerId)) this.$("box-overlay").releasePointerCapture(drawing.pointerId);
    this.$("box-overlay").replaceChildren();
  }

  cancel() {
    this.areaMode = false;
    if (this.edit) {
      const data = this.data(this.edit.ref);
      Object.assign(data, this.edit.before);
      this.edit = null;
    }
    this.cancelDrawing();
    this.pan = null;
    this.$("reference-image-viewport").classList.remove("panning");
    this.hideMagnifier();
  }

  canAdd(ref = this.ref) {
    if (this.total(ref) < this.limit(ref)) return true;
    this.options.message(this.limitMessage());
    return false;
  }

  addBox(ref, box) {
    if (this.options.isBusy() || !ref || !box.every(Number.isFinite) ||
        !(0 <= box[0] && box[0] < box[2] && box[2] <= ref.width && 0 <= box[1] && box[1] < box[3] && box[3] <= ref.height)) return;
    const data = this.data(ref), existing = data.items.findIndex((item) => item.box.every((value, index) => value === box[index]));
    if (existing !== -1 && data.items[existing].selected) {
      this.options.message("That box is already selected.");
      return;
    }
    if (!this.canAdd(ref)) return;
    this.cancel();
    this.remember(data);
    if (existing !== -1) data.items[existing].selected = true;
    else data.items.push({ box: [...box], selected: true });
    data.selectedActive = existing !== -1 ? existing : data.items.length - 1;
    this.sync(ref);
    this.options.message("");
    this.changed();
    this.announce("Drawn box selected.");
  }

  addArea(ref, box) {
    if (this.options.isBusy() || !ref || !Array.isArray(box) || box.length !== 4 || !box.every(Number.isInteger) ||
        !(0 <= box[0] && box[0] + 4 <= box[2] && box[2] <= ref.width &&
          0 <= box[1] && box[1] + 4 <= box[3] && box[3] <= ref.height)) return;
    const data = this.data(ref);
    if (data.areas.some((area) => area.every((value, index) => value === box[index]))) return;
    this.cancelDrawing();
    this.remember(data);
    data.areas.push([...box]);
    this.sync(ref);
    this.options.message("");
    this.changed();
    this.announce("Search area added. Draw another area or choose Find objects.");
  }

  removeArea(index) {
    if (!this.ref || this.options.isBusy() || this.edit) return;
    const data = this.data(this.ref);
    if (!data.areas[index]) return;
    this.remember(data);
    data.areas.splice(index, 1);
    this.sync(this.ref);
    this.changed();
    this.announce("Search area removed.");
    this.focus();
  }

  select(value = true, advance = true) {
    const ref = this.ref;
    if (!ref || this.options.isBusy() || this.edit) return;
    if (!value) { this.deselect(); return; }
    const data = this.data(ref), item = data.items[data.active];
    if (!item || item.selected || !this.canAdd(ref)) return;
    this.remember(data);
    item.selected = true;
    data.selectedActive = data.active;
    this.sync(ref);
    this.options.message("");
    this.changed();
    this.announce(`Object selected. ${this.queue(data).length} suggestions remaining.`);
  }

  deselect(index = this.ref ? this.data(this.ref).selectedActive : null) {
    if (!this.ref || this.options.isBusy() || this.edit) return;
    const data = this.data(this.ref), item = data.items[index];
    if (!item?.selected) { this.announce("Choose a selected box on the photo to deselect it."); return; }
    this.remember(data);
    item.selected = false;
    data.selectedActive = null;
    this.sync(this.ref);
    this.options.message("");
    this.changed();
    this.announce("Box deselected and returned to the suggestions.");
    this.focus();
  }

  activate(index, focus = true) {
    if (!this.ref || this.options.isBusy() || this.edit) return;
    const data = this.data(this.ref), item = data.items[index];
    if (!item) return;
    if (item.selected) data.selectedActive = index;
    else { data.active = index; data.selectedActive = null; }
    this.render(this.ref);
    if (focus) this.focus();
    this.announce(item.selected ? "Selected box focused. X deselects; E adjusts." : "Suggestion focused. Space selects.");
  }

  navigate(delta) {
    const data = this.data(this.ref), queue = this.normalize(data), position = queue.indexOf(data.active);
    if (queue.length) this.activate(queue[Math.max(0, Math.min(queue.length - 1, position + delta))]);
  }

  undo() {
    if (!this.ref || this.options.isBusy() || this.edit) return;
    const data = this.data(this.ref), before = data.history[data.history.length - 1];
    if (!before) return;
    if (this.total() - this.ref.boxes.length + before.items.filter((item) => item.selected).length > this.limit()) {
      this.options.message(this.limitMessage()); return;
    }
    data.history.pop();
    Object.assign(data, before);
    this.sync(this.ref);
    this.changed();
    this.announce("Change undone.");
    this.focus();
  }

  action(action) {
    if (!this.ref || this.options.isBusy()) return;
    const data = this.data(this.ref);
    if (action === "areas" && !this.edit) {
      this.cancelDrawing();
      this.areaMode = !this.areaMode;
      this.render(this.ref);
      this.announce(this.areaMode ? "Draw search areas, then choose Find objects." : "Draw example boxes.");
      this.focus();
      return;
    }
    if ((action === "previous" || action === "next") && !this.edit) { this.navigate(action === "next" ? 1 : -1); return; }
    if (action === "toggle" || action === "keep") this.select(true);
    if (action === "remove") this.deselect();
    if (action === "undo") this.undo();
    if ((action === "adjust" || action === "adjust-candidate") && !this.edit) {
      const index = action === "adjust-candidate" ? data.active : data.selectedActive ?? data.active;
      if (data.items[index]) {
        this.areaMode = false;
        this.cancelDrawing();
        this.edit = { ref: this.ref, index, before: this.snapshot(data) };
        if (action === "adjust-candidate") data.selectedActive = null;
        this.render(this.ref);
        this.announce("Arrows move, Shift and arrows resize. Enter saves; Escape cancels.");
      }
    }
    if (action === "save" && this.edit) {
      const box = data.items[this.edit.index].box;
      if (data.items.some((item, index) => index !== this.edit.index && item.box.every((value, coordinate) => value === box[coordinate]))) {
        const message = "That box matches an existing box. Move or resize it before saving.";
        this.options.message(message);
        this.announce(message);
        this.focus();
        return;
      }
      if (JSON.stringify(data.items) !== JSON.stringify(this.edit.before.items)) data.history.push(this.edit.before);
      this.edit = null;
      this.sync(this.ref);
      this.options.message("");
      this.changed();
      this.announce("Adjustment saved.");
    }
    if (action === "cancel" && this.edit) { this.cancel(); this.options.message(""); this.render(this.ref); this.announce("Adjustment cancelled."); }
    if (action === "done" && !this.edit && this.valid()) { this.hideMagnifier(); this.options.onDone(); return; }
    this.focus();
  }

  valid() {
    const boxes = this.ref?.boxes || [], required = this.required();
    return required == null ? !!boxes.length : this.total() === required && new Set(boxes.map((box) => JSON.stringify(box))).size === required;
  }
  focus() {
    if (this.options.isBusy()) return;
    const selected = this.ref && this.data(this.ref).selectedActive != null;
    const marker = selected && this.$("box-markers").querySelector('.reference-box-marker[aria-current="true"]');
    (marker && !marker.disabled ? marker : this.$("reference-image-viewport")).focus({ preventScroll: true });
  }

  keydown(event) {
    if (!this.ref || this.options.isBusy() || this.$("reference-editor").hidden || this.$("reference-panel").hidden ||
        event.target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])') || event.isComposing || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "escape" && this.areaMode) {
      event.preventDefault(); this.cancelDrawing(); this.areaMode = false; this.render(this.ref); this.focus(); return;
    }
    if (event.repeat && [" ", "a", "x", "delete", "backspace", "e", "enter"].includes(key)) { event.preventDefault(); return; }
    if (event.ctrlKey || event.metaKey) {
      if (key === "z" && !event.shiftKey && !this.edit) { event.preventDefault(); this.undo(); }
      return;
    }
    if (this.edit) {
      if (key === "enter" && event.target.closest('[data-reference-action],#auto-detect-boxes,#remove-reference')) return;
      if (key === "enter" || key === "escape") { event.preventDefault(); this.action(key === "enter" ? "save" : "cancel"); }
      else if (["arrowleft", "arrowright", "arrowup", "arrowdown"].includes(key)) {
        event.preventDefault();
        const box = this.data(this.ref).items[this.edit.index].box;
        const horizontal = key === "arrowleft" || key === "arrowright", change = key === "arrowleft" || key === "arrowup" ? -1 : 1;
        const start = horizontal ? 0 : 1, end = start + 2, limit = horizontal ? this.ref.width : this.ref.height;
        if (event.shiftKey) box[end] = Math.max(box[start] + Math.min(4, box[end] - box[start]), Math.min(limit, box[end] + change));
        else { const shift = Math.max(-box[start], Math.min(limit - box[end], change)); box[start] += shift; box[end] += shift; }
        this.render(this.ref);
        this.focus();
      }
      return;
    }
    if (event.shiftKey) return;
    if (key === " ") {
      event.preventDefault(); this.select(true); this.focus(); return;
    }
    const actions = { arrowleft: "previous", arrowright: "next", arrowup: "previous", arrowdown: "next", a: "keep", x: "remove", delete: "remove", backspace: "remove", e: "adjust" };
    if (actions[key]) { event.preventDefault(); this.action(actions[key]); }
  }

  crop(container, ref, box, frame) {
    const width = box[2] - box[0], height = box[3] - box[1];
    const ratio = frame ? container.clientWidth / container.clientHeight || 1 : width / height;
    const side = frame ? Math.max(width, height * ratio) * 1.45 : width;
    const x = (box[0] + box[2]) / 2 - side / 2, y = (box[1] + box[3]) / 2 - side / (2 * ratio);
    const image = container.querySelector("img");
    if (image.src !== ref.url) image.src = ref.url;
    image.style.width = `${ref.width / side * 100}%`;
    image.style.left = `${-x / side * 100}%`;
    image.style.top = `${-y / side * ratio * 100}%`;
    if (frame) {
      frame.style.left = `${(box[0] - x) / side * 100}%`; frame.style.top = `${(box[1] - y) / side * ratio * 100}%`;
      frame.style.width = `${width / side * 100}%`; frame.style.height = `${height / side * ratio * 100}%`;
    }
  }

  renderCrop() {
    if (!this.ref) return;
    const data = this.data(this.ref), item = data.items[data.active];
    if (item && !item.selected) this.crop(this.$("reference-crop"), this.ref, item.box, this.$("reference-crop-box"));
  }

  render(ref) {
    const switched = ref !== this.ref;
    if (switched || this.options.isBusy()) this.cancel();
    this.ref = ref;
    if (switched) { this.zoom = 1; this.$("reference-image-viewport").scrollLeft = this.$("reference-image-viewport").scrollTop = 0; }
    const overlay = this.$("box-overlay"), markers = this.$("box-markers"), selectedList = this.$("reference-selected-list"), areas = this.$("detection-area-markers");
    const restoreFocus = document.activeElement?.closest?.("#box-markers");
    const restoreThumbnail = document.activeElement?.closest?.("#reference-selected-list");
    if (!this.drawing) overlay.replaceChildren();
    markers.replaceChildren();
    areas.replaceChildren();
    selectedList.replaceChildren();
    selectedList.hidden = !ref?.boxes.length;
    if (!ref) { this.hideMagnifier(); return; }
    const data = this.data(ref), queue = this.normalize(data), busy = this.options.isBusy(), total = this.total(ref), limit = this.limit(ref);
    overlay.setAttribute("viewBox", `0 0 ${ref.width} ${ref.height}`);
    overlay.setAttribute("preserveAspectRatio", "none");
    const highlighted = this.edit?.index ?? data.selectedActive ?? data.active;
    data.items.forEach((item, index) => {
      const group = document.createElement("div"); group.className = "reference-box-group";
      const [x1, y1, x2, y2] = item.box;
      group.style.left = `${x1 / ref.width * 100}%`; group.style.top = `${y1 / ref.height * 100}%`;
      group.style.width = `${(x2 - x1) / ref.width * 100}%`; group.style.height = `${(y2 - y1) / ref.height * 100}%`;
      const button = document.createElement("button"); button.type = "button"; button.className = item.selected ? "reference-box-marker" : "reference-proposal-marker"; button.dataset.index = String(index);
      // Keep geometry on the marker as well as its group for accessible/test inspection.
      button.dataset.box = JSON.stringify(item.box);
      button.setAttribute("aria-pressed", String(item.selected)); button.setAttribute("aria-current", String(index === highlighted));
      button.setAttribute("aria-label", item.selected ? `Selected box ${index + 1}. Press X to deselect or E to adjust.` : `Suggestion ${index + 1}. Press Space to select.`);
      button.disabled = busy || (!!this.edit && index !== this.edit.index);
      button.classList.toggle("adjusting", !!this.edit && index === this.edit.index);
      button.addEventListener("click", () => { if (!this.areaMode && !this.suppressMarkerClick) this.activate(index); });
      button.addEventListener("focus", () => {
        const current = this.data(ref);
        if (item.selected ? current.selectedActive !== index : current.active !== index || current.selectedActive != null) this.activate(index, false);
      });
      button.addEventListener("pointerdown", (event) => {
        if (this.edit || this.options.isBusy() || event.button !== 0) return;
        this.suppressMarkerClick = false;
        if (!this.areaMode) this.activate(index);
        this.startDrawing(event, true);
      });
      group.append(button); markers.append(group);
      if (!item.selected) return;
      const entry = document.createElement("div"); entry.className = "reference-selected-item";
      const scale = 72 / Math.max(x2 - x1, y2 - y1);
      entry.style.width = `${(x2 - x1) * scale}px`; entry.style.height = `${(y2 - y1) * scale}px`;
      const thumbnail = document.createElement("button"); thumbnail.type = "button"; thumbnail.className = "reference-selected-thumbnail";
      thumbnail.dataset.index = String(index); thumbnail.dataset.box = JSON.stringify(item.box);
      thumbnail.setAttribute("aria-label", `Selected example ${index + 1}. Press X to deselect.`);
      thumbnail.setAttribute("aria-current", String(index === data.selectedActive));
      thumbnail.disabled = busy || !!this.edit;
      const image = document.createElement("img"); image.alt = ""; image.draggable = false;
      thumbnail.append(image);
      thumbnail.addEventListener("click", () => this.activate(index));
      thumbnail.addEventListener("focus", () => { if (this.data(ref).selectedActive !== index) this.activate(index, false); });
      const remove = document.createElement("button"); remove.type = "button"; remove.className = "reference-box-remove"; remove.dataset.index = String(index);
      remove.textContent = "\u00d7"; remove.setAttribute("aria-label", `Deselect box ${index + 1}`); remove.title = "Deselect (X)";
      remove.disabled = busy || !!this.edit;
      remove.addEventListener("pointerdown", (event) => event.stopPropagation());
      remove.addEventListener("click", (event) => { event.stopPropagation(); this.deselect(index); });
      entry.append(thumbnail, remove); selectedList.append(entry);
      this.crop(thumbnail, ref, item.box);
    });
    data.areas.forEach((box, index) => {
      const group = document.createElement("div"); group.className = "reference-detection-area"; group.dataset.box = JSON.stringify(box);
      const [x1, y1, x2, y2] = box;
      group.style.left = `${x1 / ref.width * 100}%`; group.style.top = `${y1 / ref.height * 100}%`;
      group.style.width = `${(x2 - x1) / ref.width * 100}%`; group.style.height = `${(y2 - y1) / ref.height * 100}%`;
      const remove = document.createElement("button"); remove.type = "button"; remove.className = "reference-detection-area-remove";
      remove.textContent = "×"; remove.setAttribute("aria-label", `Remove search area ${index + 1}`);
      remove.disabled = busy || !!this.edit;
      remove.addEventListener("pointerdown", (event) => event.stopPropagation());
      remove.addEventListener("click", () => this.removeArea(index));
      group.append(remove); areas.append(group);
    });
    const hasItem = data.active !== -1, selected = data.selectedActive != null, position = queue.indexOf(data.active), editing = !!this.edit;
    this.$("reference-inspector").hidden = !hasItem;
    this.$("reference-no-proposals").hidden = hasItem;
    this.$("reference-box-position").textContent = hasItem ? `${position + 1} / ${queue.length}` : "";
    this.$("reference-crop").dataset.index = String(data.active);
    this.$("reference-crop").dataset.box = hasItem ? JSON.stringify(data.items[data.active].box) : "";
    this.$("reference-crop").setAttribute("aria-label", hasItem ? `Suggestion ${position + 1} of ${queue.length}` : "No more suggestions");
    this.$("reference-box-prev").disabled = busy || editing || !hasItem || position === 0;
    this.$("reference-box-next").disabled = busy || editing || !hasItem || position === queue.length - 1;
    this.$("reference-box-toggle").disabled = busy || editing || !hasItem || total >= limit;
    this.$("reference-box-adjust").disabled = busy || editing || !hasItem;
    this.$("reference-selected-actions").hidden = !selected || editing;
    this.$("reference-selected-adjust").disabled = this.$("reference-box-remove").disabled = busy || !selected;
    this.$("reference-box-actions").hidden = editing;
    this.$("reference-box-edit-actions").hidden = !editing;
    this.$("reference-key-hints").hidden = editing;
    this.$("reference-edit-hints").hidden = !editing;
    this.$("reference-box-save").disabled = this.$("reference-box-cancel").disabled = busy;
    this.$("reference-done").disabled = busy || editing || !this.valid();
    this.$("undo-box").disabled = busy || editing || !data.history.length;
    this.$("add-coordinate-box").disabled = busy || editing || total >= limit;
    this.$("reference-select-areas").disabled = busy || editing;
    this.$("reference-select-areas").setAttribute("aria-pressed", String(this.areaMode));
    this.$("reference-select-areas").textContent = this.areaMode ? "Done marking" : "Mark areas";
    this.$("reference-no-proposals").textContent = this.areaMode ? "Draw areas, then choose Find objects or Quick count." :
      "Draw around examples, or mark areas to find objects.";
    this.$("auto-detect-boxes").disabled = busy || editing || !data.areas.length;
    this.$("auto-detect-boxes").setAttribute("aria-busy", String(this.options.isDetecting(ref)));
    this.$("drawing-surface").classList.toggle("detecting", this.options.isDetecting(ref));
    this.$("reference-selected-count").textContent = `${total} / ${this.required(ref) ?? limit} selected`;
    this.layoutImage();
    this.renderCrop();
    if (restoreThumbnail && !busy) {
      const thumbnail = selectedList.querySelector('.reference-selected-thumbnail[aria-current="true"]');
      if (thumbnail && !thumbnail.disabled) thumbnail.focus({ preventScroll: true });
      else this.focus();
    } else if (restoreFocus && !busy) this.focus();
    if (busy || this.$("reference-editor").hidden) this.hideMagnifier();
  }

  layoutImage() {
    if (!this.ref) return;
    const viewport = this.$("reference-image-viewport"), stage = this.$("reference-image-stage"), surface = this.$("drawing-surface");
    if (!viewport.clientWidth || !viewport.clientHeight) return;
    this.fitScale = Math.min(viewport.clientWidth / this.ref.width, viewport.clientHeight / this.ref.height);
    const width = this.ref.width * this.fitScale * this.zoom, height = this.ref.height * this.fitScale * this.zoom;
    const stageWidth = Math.max(viewport.clientWidth, width), stageHeight = Math.max(viewport.clientHeight, height);
    stage.style.width = `${stageWidth}px`; stage.style.height = `${stageHeight}px`;
    surface.style.width = `${width}px`; surface.style.height = `${height}px`;
    surface.style.left = `${(stageWidth - width) / 2}px`; surface.style.top = `${(stageHeight - height) / 2}px`;
    this.$("reference-zoom-reset").textContent = `${Math.round(this.zoom * 100)}%`;
    this.$("reference-zoom-reset").setAttribute("aria-label", `Zoom ${Math.round(this.zoom * 100)} percent. Reset to fit photo.`);
    this.$("reference-zoom-in").disabled = this.options.isBusy() || this.zoom >= 16;
    this.$("reference-zoom-out").disabled = this.options.isBusy() || this.zoom <= 1;
    this.$("reference-zoom-reset").disabled = this.options.isBusy();
  }

  setZoom(value, clientX, clientY) {
    if (!this.ref || this.options.isBusy() || this.drawing) return;
    const viewport = this.$("reference-image-viewport"), rect = viewport.getBoundingClientRect(), image = this.$("drawing-surface").getBoundingClientRect();
    const x = clientX ?? rect.left + viewport.clientWidth / 2, y = clientY ?? rect.top + viewport.clientHeight / 2;
    const fx = Math.max(0, Math.min(1, (x - image.left) / image.width)), fy = Math.max(0, Math.min(1, (y - image.top) / image.height));
    this.zoom = Math.max(1, Math.min(16, value));
    this.layoutImage();
    const width = this.ref.width * this.fitScale * this.zoom, height = this.ref.height * this.fitScale * this.zoom;
    viewport.scrollLeft = Math.max(0, (Math.max(viewport.clientWidth, width) - width) / 2 + fx * width - (x - rect.left));
    viewport.scrollTop = Math.max(0, (Math.max(viewport.clientHeight, height) - height) / 2 + fy * height - (y - rect.top));
    this.hideMagnifier();
  }

  hideMagnifier() { this.$("reference-magnifier").hidden = true; }
  magnify(event) {
    const ref = this.ref, image = this.$("editor-image"), lens = this.$("reference-magnifier");
    // A native source pixel per CSS pixel is enough detail. Intrinsically small
    // images already enlarged at fit get a useful lens until 2.5 times fit.
    if (!ref || this.options.isBusy() || this.pan || event.pointerType === "touch" || !image.complete || !image.naturalWidth ||
        this.$("reference-editor").hidden || (this.fitScale < 1 ? this.fitScale * this.zoom >= 1 : this.zoom >= 2.5)) { this.hideMagnifier(); return; }
    const rect = image.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) { this.hideMagnifier(); return; }
    const size = 144, ratio = globalThis.devicePixelRatio || 1, magnification = Math.max(1.5, rect.width / ref.width * 2.5) * this.magnifierZoom;
    const x = (event.clientX - rect.left) / rect.width * image.naturalWidth, y = (event.clientY - rect.top) / rect.height * image.naturalHeight;
    const sourceWidth = size / magnification * image.naturalWidth / ref.width, sourceHeight = size / magnification * image.naturalHeight / ref.height;
    const context = lens.getContext("2d");
    lens.width = Math.round(size * ratio); lens.height = Math.round(size * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = "#e8eeec"; context.fillRect(0, 0, size, size);
    context.drawImage(image, x - sourceWidth / 2, y - sourceHeight / 2, sourceWidth, sourceHeight, 0, 0, size, size);
    context.setLineDash([1, 5]); context.lineCap = "round";
    context.beginPath();
    context.moveTo(0, size / 2); context.lineTo(size, size / 2);
    context.moveTo(size / 2, 0); context.lineTo(size / 2, size);
    context.strokeStyle = "#142031b3"; context.lineWidth = 3; context.stroke();
    context.strokeStyle = "#ffffff"; context.lineWidth = 1.5; context.stroke();
    const windowWidth = globalThis.innerWidth || document.documentElement.clientWidth, windowHeight = globalThis.innerHeight || document.documentElement.clientHeight;
    let left = event.clientX + 20, top = event.clientY + 20;
    if (left + size + 8 > windowWidth) left = event.clientX - size - 20;
    if (top + size + 8 > windowHeight) top = event.clientY - size - 20;
    lens.style.left = `${Math.max(8, Math.min(windowWidth - size - 8, left))}px`;
    lens.style.top = `${Math.max(8, Math.min(windowHeight - size - 8, top))}px`;
    lens.hidden = false;
  }

  point(event, ref) {
    const bounds = this.$("box-overlay").getBoundingClientRect();
    return [Math.round(Math.max(0, Math.min(ref.width, (event.clientX - bounds.left) / bounds.width * ref.width))),
      Math.round(Math.max(0, Math.min(ref.height, (event.clientY - bounds.top) / bounds.height * ref.height)))];
  }
  startDrawing(event, fromMarker = false) {
    if (!this.ref || this.edit || this.options.isBusy() || event.button !== 0) return;
    event.preventDefault();
    this.$("box-overlay").setPointerCapture(event.pointerId);
    this.drawing = { ref: this.ref, pointerId: event.pointerId, start: this.point(event, this.ref), clientStart: [event.clientX, event.clientY], fromMarker, area: this.areaMode, box: null };
  }
  draw(event) {
    if (!this.drawing || event.pointerId !== this.drawing.pointerId || this.drawing.ref !== this.ref) return;
    if (Math.hypot(event.clientX - this.drawing.clientStart[0], event.clientY - this.drawing.clientStart[1]) < 4) {
      this.drawing.box = null; this.$("box-overlay").replaceChildren(); return;
    }
    const [x, y] = this.point(event, this.ref), [sx, sy] = this.drawing.start;
    const box = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)];
    this.drawing.box = box;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.classList.add(this.drawing.area ? "area-draft" : "draft"); rect.setAttribute("x", box[0]); rect.setAttribute("y", box[1]);
    rect.setAttribute("width", box[2] - box[0]); rect.setAttribute("height", box[3] - box[1]);
    this.$("box-overlay").replaceChildren(rect);
  }
  finishDrawing(event) {
    if (!this.drawing || event.pointerId !== this.drawing.pointerId) return;
    this.draw(event);
    const { ref, box, fromMarker, area } = this.drawing;
    this.suppressMarkerClick = fromMarker && !!box;
    this.cancelDrawing();
    if (ref === this.ref && box && box[2] - box[0] >= 4 && box[3] - box[1] >= 4) {
      if (area) this.addArea(ref, box);
      else this.addBox(ref, box);
    }
    this.render(this.ref);
  }
};
