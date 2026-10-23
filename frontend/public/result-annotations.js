(() => {
  const ns = "http://www.w3.org/2000/svg";
  const sizeFor = (result) => Math.max(12, Math.min(28, Math.round(Math.min(result.width, result.height) / 55)));
  const lineFor = (result) => Math.max(2, Math.round(Math.min(result.width, result.height) / 400));

  function render(result, url) {
    const svg = document.createElementNS(ns, "svg");
    svg.classList.add("annotated-image");
    svg.setAttribute("viewBox", `0 0 ${result.width} ${result.height}`);
    svg.setAttribute("width", result.width);
    svg.setAttribute("height", result.height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `Annotated count for ${result.filename}: ${result.count} objects`);
    const image = document.createElementNS(ns, "image");
    image.setAttribute("href", url);
    image.setAttribute("width", result.width);
    image.setAttribute("height", result.height);
    svg.append(image);
    for (const detection of result.detections || []) {
      const [left, top, right, bottom] = detection.box;
      const rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", left);
      rect.setAttribute("y", top);
      rect.setAttribute("width", right - left);
      rect.setAttribute("height", bottom - top);
      rect.setAttribute("stroke-width", lineFor(result));
      svg.append(rect);
    }
    (result.detections || []).forEach((detection, index) => {
      const [left, top, right, bottom] = detection.box;
      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", Math.max(sizeFor(result), Math.min(result.width - sizeFor(result), (left + right) / 2)));
      label.setAttribute("y", Math.max(sizeFor(result), Math.min(result.height - sizeFor(result), (top + bottom) / 2)));
      label.setAttribute("font-size", sizeFor(result));
      label.textContent = String(index + 1);
      svg.append(label);
    });
    const badge = document.createElementNS(ns, "g");
    badge.classList.add("count-badge");
    const background = document.createElementNS(ns, "rect");
    background.setAttribute("width", Math.max(100, (`COUNT: ${result.count}`.length * sizeFor(result) * .7) + 24));
    background.setAttribute("height", sizeFor(result) + 20);
    badge.append(background);
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", 12);
    text.setAttribute("y", sizeFor(result) + 8);
    text.setAttribute("font-size", sizeFor(result));
    text.textContent = `COUNT: ${result.count}`;
    badge.append(text);
    svg.append(badge);
    return svg;
  }

  async function download(result, url) {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = result.width;
    canvas.height = result.height;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0, result.width, result.height);
    context.strokeStyle = "#34e3aa";
    context.lineWidth = lineFor(result);
    for (const detection of result.detections || []) {
      const [left, top, right, bottom] = detection.box;
      context.strokeRect(left, top, right - left, bottom - top);
    }
    const size = sizeFor(result);
    context.font = `bold ${size}px sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    (result.detections || []).forEach((detection, index) => {
      const [left, top, right, bottom] = detection.box;
      const x = Math.max(size, Math.min(result.width - size, (left + right) / 2));
      const y = Math.max(size, Math.min(result.height - size, (top + bottom) / 2));
      const label = String(index + 1);
      const labelWidth = context.measureText(label).width + 12;
      context.fillStyle = "#073d32";
      context.fillRect(x - labelWidth / 2, y - size / 2 - 4, labelWidth, size + 8);
      context.fillStyle = "white";
      context.fillText(label, x, y);
    });
    context.textAlign = "left";
    context.textBaseline = "alphabetic";
    const countLabel = `COUNT: ${result.count}`;
    context.fillStyle = "#102522";
    context.fillRect(0, 0, context.measureText(countLabel).width + 24, size + 20);
    context.fillStyle = "white";
    context.fillText(countLabel, 12, size + 8);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", .94));
    if (!blob) throw new Error("The image could not be prepared.");
    const link = document.createElement("a");
    const downloadUrl = URL.createObjectURL(blob);
    link.href = downloadUrl;
    link.download = `${result.filename.replace(/\.[^.]+$/, "") || "count"}-counted.jpg`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(downloadUrl), 30_000);
  }

  globalThis.ResultAnnotations = { render, download };
})();
