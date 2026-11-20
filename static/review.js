(function () {
  'use strict';

  const C = window.REVIEW;
  const SIZE = 224;
  const VIEW = 448;
  const K = VIEW / SIZE;
  const MIN_AREA = 15;
  const KEYS = ['data', 'result', 'findings', 'limits', 'doctor'];

  const $ = (id) => document.getElementById(id);
  const canvas = $('view');
  const ctx = canvas.getContext('2d');

  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = maskCanvas.height = SIZE;
  const mctx = maskCanvas.getContext('2d', { willReadFrequently: true });

  const hlCanvas = document.createElement('canvas');
  hlCanvas.width = hlCanvas.height = SIZE;
  const hctx = hlCanvas.getContext('2d');

  const baseImg = new Image();
  let regions = [];
  let labelMap = new Int32Array(SIZE * SIZE);          // номер области в каждом пикселе, 0 = фон
  let highlight = 0;
  let tool = 'brush';
  let drawing = false;
  let last = null;
  let dirty = false;
  let findingsEdited = false;

  // ---------- загрузка ----------
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error('не загрузилось: ' + url));
      im.src = url;
    });
  }

  async function setMask(url) {
    mctx.clearRect(0, 0, SIZE, SIZE);
    if (!url) return;
    try {
      const im = await loadImage(url);
      const tmp = document.createElement('canvas');
      tmp.width = tmp.height = SIZE;
      const tc = tmp.getContext('2d');
      tc.drawImage(im, 0, 0, SIZE, SIZE);
      const src = tc.getImageData(0, 0, SIZE, SIZE).data;
      const out = mctx.createImageData(SIZE, SIZE);
      for (let i = 0; i < SIZE * SIZE; i++) {
        out.data[i * 4] = 255;
        out.data[i * 4 + 3] = src[i * 4] > 127 ? 255 : 0;
      }
      mctx.putImageData(out, 0, 0);
    } catch (e) {
      // маски нет: остаётся пустая
    }
  }

  // ---------- области и измерения ----------
  function analyze() {
    const data = mctx.getImageData(0, 0, SIZE, SIZE).data;
    const raw = new Int32Array(SIZE * SIZE);
    const found = [];

    for (let s = 0; s < SIZE * SIZE; s++) {
      if (data[s * 4 + 3] <= 127 || raw[s]) continue;
      const id = found.length + 1;
      raw[s] = id;
      const stack = [s];
      let area = 0, sx = 0, sy = 0, x0 = SIZE, y0 = SIZE, x1 = 0, y1 = 0;

      while (stack.length) {
        const p = stack.pop();
        const x = p % SIZE;
        const y = (p - x) / SIZE;
        area++; sx += x; sy += y;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;

        const nb = [];
        if (x > 0) nb.push(p - 1);
        if (x < SIZE - 1) nb.push(p + 1);
        if (y > 0) nb.push(p - SIZE);
        if (y < SIZE - 1) nb.push(p + SIZE);
        for (const q of nb) {
          if (!raw[q] && data[q * 4 + 3] > 127) {
            raw[q] = id;
            stack.push(q);
          }
        }
      }
      found.push({ id, area, cx: sx / area, cy: sy / area, bbox: [x0, y0, x1, y1] });
    }

    const kept = found.filter((r) => r.area >= MIN_AREA).sort((a, b) => b.area - a.area);
    const numberOf = {};
    kept.forEach((r, i) => { r.n = i + 1; numberOf[r.id] = r.n; });

    labelMap = new Int32Array(SIZE * SIZE);
    for (let i = 0; i < raw.length; i++) {
      if (raw[i] && numberOf[raw[i]]) labelMap[i] = numberOf[raw[i]];
    }
    regions = kept;
  }

  function position(r) {
    const [hx0, hy0, hx1, hy1] = C.headBbox;
    const hw = hx1 - hx0 + 1;
    const hh = hy1 - hy0 + 1;
    const dx = (r.cx - (hx0 + hw / 2)) / (hw / 2);
    const dy = (r.cy - (hy0 + hh / 2)) / (hh / 2);
    const h = Math.abs(dx) < 0.25 ? '' : (dx > 0 ? 'правой' : 'левой');
    const v = Math.abs(dy) < 0.25 ? '' : (dy > 0 ? 'нижней' : 'верхней');
    if (!h && !v) return 'в центральной части изображения';
    if (h && v) return `в ${v} ${h} части изображения`;
    return `в ${h || v} части изображения`;
  }

  function sentence(r) {
    const [x0, y0, x1, y1] = r.bbox;
    const size = Math.max(x1 - x0 + 1, y1 - y0 + 1);
    const pct = Math.max(0.1, (100 * r.area) / (C.headArea || SIZE * SIZE)).toFixed(1).replace('.', ',');
    return `${position(r)}; площадь около ${r.area} пикселей (~${pct}% площади головы). `;
  }

  function findingsText() {
    if (!regions.length) {
      return 'Областей, подозрительных на опухоль, автоматически не выделено.';
    }
    const lines = regions.map((r) => `Область ${r.n}: ${sentence(r)}`);
    return lines.join('\n');
  }

  // ---------- отрисовка ----------
  function render(forExport) {
    ctx.clearRect(0, 0, VIEW, VIEW);
    ctx.imageSmoothingEnabled = true;
    if (baseImg.complete && baseImg.naturalWidth) ctx.drawImage(baseImg, 0, 0, VIEW, VIEW);

    if (!forExport && !$('show-mask').checked) return;

    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = 0.45;
    ctx.drawImage(maskCanvas, 0, 0, VIEW, VIEW);
    ctx.restore();

    if (!forExport && highlight) {
      const img = hctx.createImageData(SIZE, SIZE);
      for (let i = 0; i < SIZE * SIZE; i++) {
        if (labelMap[i] === highlight) {
          img.data[i * 4] = 250; img.data[i * 4 + 1] = 204; img.data[i * 4 + 2] = 21; img.data[i * 4 + 3] = 170;
        }
      }
      hctx.putImageData(img, 0, 0);
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(hlCanvas, 0, 0, VIEW, VIEW);
      ctx.restore();
    }

    regions.forEach((r) => {
      const [x0, y0, x1, y1] = r.bbox;
      const hot = !forExport && r.n === highlight;
      ctx.lineWidth = hot ? 3 : 1.5;
      ctx.strokeStyle = hot ? '#facc15' : '#ef4444';
      ctx.strokeRect(x0 * K, y0 * K, (x1 - x0 + 1) * K, (y1 - y0 + 1) * K);
      ctx.fillStyle = hot ? '#facc15' : '#ef4444';
      ctx.font = 'bold 16px sans-serif';
      ctx.fillText(String(r.n), x0 * K + 3, Math.max(14, y0 * K - 4));
    });
  }

  function updateList() {
    const ul = $('regions');
    ul.innerHTML = '';
    if (!regions.length) {
      const li = document.createElement('li');
      li.className = 'muted';
      li.textContent = 'Областей нет. Выделите область кистью на снимке, если видите образование.';
      ul.appendChild(li);
      return;
    }
    regions.forEach((r) => {
      const li = document.createElement('li');
      li.dataset.n = String(r.n);
      li.textContent = `Область ${r.n}: ${sentence(r)}`;
      li.addEventListener('mouseenter', () => setHighlight(r.n));
      li.addEventListener('mouseleave', () => setHighlight(0));
      ul.appendChild(li);
    });
    markList();
  }

  function markList() {
    document.querySelectorAll('#regions li').forEach((li) => {
      li.classList.toggle('hot', Number(li.dataset.n) === highlight);
    });
  }

  function setHighlight(n) {
    if (n === highlight) return;
    highlight = n;
    markList();
    render();
  }

  // ---------- статус ----------
  function setStatus(text, cls) {
    const s = $('status');
    s.textContent = text;
    s.className = 'status' + (cls ? ' ' + cls : '');
  }

  function markDirty() {
    dirty = true;
    setStatus('Есть несохранённые изменения.');
  }

  function showDownloads(htmlUrl, txtUrl) {
    if (!htmlUrl) return;
    const stamp = Date.now();
    $('dl-html').href = htmlUrl + '?v=' + stamp;
    $('dl-html').setAttribute('download', C.stem + '_conclusion_final.html');
    $('dl-txt').href = txtUrl + '?v=' + stamp;
    $('dl-txt').setAttribute('download', C.stem + '_conclusion_final.txt');
    $('downloads').hidden = false;
  }

  // ---------- рисование ----------
  function toMask(e) {
    const rect = canvas.getBoundingClientRect();
    return [(e.clientX - rect.left) * SIZE / rect.width, (e.clientY - rect.top) * SIZE / rect.height];
  }

  function paint(a, b) {
    const r = Number($('brush-size').value);
    mctx.save();
    mctx.globalCompositeOperation = tool === 'brush' ? 'source-over' : 'destination-out';
    mctx.strokeStyle = 'rgb(255,0,0)';
    mctx.fillStyle = 'rgb(255,0,0)';
    mctx.lineWidth = 2 * r;
    mctx.lineCap = 'round';
    mctx.beginPath();
    mctx.moveTo(a[0], a[1]);
    mctx.lineTo(b[0], b[1]);
    mctx.stroke();
    mctx.beginPath();
    mctx.arc(b[0], b[1], r, 0, Math.PI * 2);
    mctx.fill();
    mctx.restore();
    render();
  }

  function maskChanged() {
    analyze();
    updateList();
    if (!findingsEdited) $('sec-findings').value = findingsText();
    render();
    markDirty();
  }

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    drawing = true;
    last = toMask(e);
    paint(last, last);
    e.preventDefault();
  });

  canvas.addEventListener('pointermove', (e) => {
    const p = toMask(e);
    if (drawing) {
      paint(last, p);
      last = p;
    } else {
      const x = Math.floor(p[0]);
      const y = Math.floor(p[1]);
      const n = (x >= 0 && y >= 0 && x < SIZE && y < SIZE) ? labelMap[y * SIZE + x] : 0;
      setHighlight(n);
    }
  });

  function stopDrawing() {
    if (!drawing) return;
    drawing = false;
    maskChanged();
  }
  canvas.addEventListener('pointerup', stopDrawing);
  canvas.addEventListener('pointercancel', stopDrawing);
  canvas.addEventListener('pointerleave', () => { if (!drawing) setHighlight(0); });

  // ---------- кнопки ----------
  function selectTool(name) {
    tool = name;
    $('tool-brush').classList.toggle('active', name === 'brush');
    $('tool-eraser').classList.toggle('active', name === 'eraser');
  }
  $('tool-brush').addEventListener('click', () => selectTool('brush'));
  $('tool-eraser').addEventListener('click', () => selectTool('eraser'));
  $('show-mask').addEventListener('change', () => render());

  $('reset-mask').addEventListener('click', async () => {
    await setMask(C.modelMaskUrl);
    maskChanged();
  });
  $('clear-mask').addEventListener('click', () => {
    mctx.clearRect(0, 0, SIZE, SIZE);
    maskChanged();
  });

  KEYS.forEach((k) => {
    $('sec-' + k).addEventListener('input', () => {
      if (k === 'findings') findingsEdited = true;
      markDirty();
    });
  });
  $('doctor').addEventListener('input', markDirty);

  // ---------- сохранение ----------
  function maskDataUrl() {
    const exp = document.createElement('canvas');
    exp.width = exp.height = SIZE;
    const ec = exp.getContext('2d');
    const src = mctx.getImageData(0, 0, SIZE, SIZE).data;
    const out = ec.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const v = src[i * 4 + 3] > 127 ? 255 : 0;
      out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = v;
      out.data[i * 4 + 3] = 255;
    }
    ec.putImageData(out, 0, 0);
    return exp.toDataURL('image/png');
  }

  $('save').addEventListener('click', async () => {
    const button = $('save');
    button.disabled = true;
    setStatus('Сохраняю…');
    try {
      analyze();
      const sections = {};
      KEYS.forEach((k) => { sections[k] = $('sec-' + k).value; });

      render(true);
      const annotated = canvas.toDataURL('image/png');
      render();

      const res = await fetch(C.saveUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sections, doctor: $('doctor').value, mask: maskDataUrl(), annotated }),
      });
      const j = await res.json();
      if (!res.ok || !j.ok) throw new Error(j.error || 'ошибка сервера');

      dirty = false;
      setStatus('Сохранено ' + j.saved_at + '.', 'ok');
      showDownloads(j.html_url, j.txt_url);
    } catch (e) {
      setStatus('Не удалось сохранить: ' + e.message, 'err');
    } finally {
      button.disabled = false;
    }
  });

  window.addEventListener('beforeunload', (e) => {
    if (dirty) {
      e.preventDefault();
      e.returnValue = '';
    }
  });

  // ---------- запуск ----------
  async function init() {
    KEYS.forEach((k) => { $('sec-' + k).value = (C.sections && C.sections[k]) || ''; });
    $('doctor').value = C.doctor || '';

    try {
      const im = await loadImage(C.viewUrl);
      baseImg.src = im.src;
      await new Promise((resolve) => { baseImg.onload = resolve; baseImg.onerror = resolve; if (baseImg.complete) resolve(); });
    } catch (e) {
      setStatus('Не удалось загрузить снимок.', 'err');
    }

    await setMask(C.maskUrl);
    analyze();
    updateList();

    findingsEdited = Boolean(C.findingsSaved);
    if (!C.findingsSaved && !$('sec-findings').value.trim()) {
      $('sec-findings').value = findingsText();
    }
    render();

    if (C.savedAt) {
      setStatus('Сохранено ' + C.savedAt + '.', 'ok');
      showDownloads(C.finalHtmlUrl, C.finalTxtUrl);
    }
    dirty = false;
  }

  init();
})();
