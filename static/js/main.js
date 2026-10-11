/* SmartCrop AI – main.js */

/* ── Mobile nav toggle ── */
const toggle = document.getElementById("navToggle");
const links  = document.getElementById("navLinks");
if (toggle && links) {
  toggle.addEventListener("click", () => links.classList.toggle("open"));
}

/* ── Client-side form validation ── */
const form = document.getElementById("recommendForm");
if (form) {
  const RANGES = {
    N:           [0,   200],
    P:           [0,   200],
    K:           [0,   200],
    temperature: [-10,  55],
    humidity:    [0,   100],
    ph:          [0,    14],
    rainfall:    [0,  4000],
  };
  const LABELS = {
    N: "Nitrogen", P: "Phosphorus", K: "Potassium",
    temperature: "Temperature", humidity: "Humidity",
    ph: "pH", rainfall: "Rainfall",
  };

  form.addEventListener("submit", function (e) {
    let firstError = null;
    // clear previous errors
    form.querySelectorAll(".client-error").forEach(el => el.remove());
    form.querySelectorAll(".input-error").forEach(el => el.classList.remove("input-error"));

    for (const [field, [lo, hi]] of Object.entries(RANGES)) {
      const input = form.querySelector(`[name="${field}"]`);
      if (!input) continue;
      const val = input.value.trim();
      let msg = null;
      if (val === "") {
        msg = `${LABELS[field]} is required.`;
      } else if (isNaN(Number(val))) {
        msg = `${LABELS[field]} must be a number.`;
      } else {
        const num = Number(val);
        if (num < lo || num > hi) msg = `${LABELS[field]} must be between ${lo} and ${hi}.`;
      }
      if (msg) {
        input.classList.add("input-error");
        const span = document.createElement("span");
        span.className = "form-hint client-error";
        span.style.color = "#c0392b";
        span.textContent = msg;
        input.parentNode.appendChild(span);
        if (!firstError) firstError = input;
      }
    }
    if (firstError) {
      e.preventDefault();
      firstError.focus();
      firstError.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
}

/* ── Model comparison bar chart ── */
if (typeof window.CHART_DATA !== "undefined") {
  const canvas = document.getElementById("modelChart");
  if (canvas) {
    drawBarChart(canvas, window.CHART_DATA);
  }
}

function drawBarChart(canvas, data) {
  const ctx    = canvas.getContext("2d");
  const dpr    = window.devicePixelRatio || 1;
  const W      = canvas.offsetWidth  || 700;
  const H      = canvas.offsetHeight || 260;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  const labels  = data.labels;
  const f1vals  = data.f1;
  const accvals = data.acc;
  const n       = labels.length;

  const pad     = { top: 30, right: 30, bottom: 50, left: 50 };
  const chartW  = W - pad.left - pad.right;
  const chartH  = H - pad.top  - pad.bottom;

  const groupW  = chartW / n;
  const barW    = groupW * 0.3;
  const gap     = groupW * 0.05;
  const COLORS  = ["#2d9b55", "#8b6914", "#1a5fa0"];
  const series  = [
    { label: "Accuracy", values: accvals, color: COLORS[2] },
    { label: "F1-Score", values: f1vals,  color: COLORS[0] },
  ];

  // background
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);

  // horizontal grid lines
  for (let i = 0; i <= 5; i++) {
    const yv   = i / 5;
    const yPos = pad.top + chartH * (1 - yv);
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, yPos);
    ctx.lineTo(pad.left + chartW, yPos);
    ctx.stroke();
    // y-axis labels
    ctx.fillStyle   = "#5a6b5a";
    ctx.font        = "11px Inter, sans-serif";
    ctx.textAlign   = "right";
    ctx.fillText(yv.toFixed(1), pad.left - 6, yPos + 4);
  }

  // bars
  series.forEach((s, si) => {
    s.values.forEach((val, gi) => {
      const totalBarWidth = series.length * barW + (series.length - 1) * gap;
      const groupStart    = pad.left + gi * groupW + (groupW - totalBarWidth) / 2;
      const x             = groupStart + si * (barW + gap);
      const barH          = chartH * val;
      const y             = pad.top + chartH - barH;

      ctx.fillStyle = s.color;
      roundRect(ctx, x, y, barW, barH, 3);

      // value label on bar
      ctx.fillStyle   = s.color;
      ctx.font        = "bold 11px Inter, sans-serif";
      ctx.textAlign   = "center";
      ctx.fillText(val.toFixed(3), x + barW / 2, y - 4);
    });
  });

  // x-axis labels
  labels.forEach((lbl, gi) => {
    const cx = pad.left + gi * groupW + groupW / 2;
    ctx.fillStyle = "#1a2e1a";
    ctx.font      = "12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(lbl, cx, H - 12);
  });

  // legend
  const legX = W - pad.right - 160;
  series.forEach((s, i) => {
    const lx = legX + i * 85;
    ctx.fillStyle = s.color;
    roundRect(ctx, lx, 10, 14, 14, 2);
    ctx.fillStyle = "#1a2e1a";
    ctx.font      = "11px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(s.label, lx + 18, 22);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fill();
}
