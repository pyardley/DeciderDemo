const POLICY =
  "Duplicate charges are eligible for a refund. Delivery delays are not refundable; send a status update. Login lockouts go to account access, not billing.";

const SCENARIOS = [
  {
    id: "duplicate",
    label: "Duplicate charge",
    ticket:
      "I was charged twice for order A-104. Please refund the duplicate. The second charge posted this morning and I need it back before rent is due.",
  },
  {
    id: "crash",
    label: "App crash",
    ticket:
      "The iOS app crashes every time I open a PDF attachment. This started after yesterday's update. Android still works. I do not need a refund.",
  },
  {
    id: "lockout",
    label: "Locked out",
    ticket:
      "I cannot sign in. It says the account is locked after too many attempts. I have a meeting in an hour and need the shared drive. Please do not refund anything.",
  },
];

const QUESTIONS = {
  department: {
    type: "choice",
    instructions: "Which team should handle this?",
    criteria: {
      billing: "Charges, invoices, refunds and duplicate payments",
      technical: "Bugs, crashes and outages",
      account: "Login, lockouts and access",
    },
  },
  refund: {
    type: "noul",
    instructions: "Is the customer asking for a refund?",
  },
  urgency: {
    type: "score",
    instructions: "How urgently does a person need to step in?",
    criteria: ["can wait", "reply today", "drop everything"],
  },
};

const ticket = document.querySelector("#ticket");
const policy = document.querySelector("#policy");
const scenarios = document.querySelector("#scenarios");
const ask = document.querySelector("#ask");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const meta = document.querySelector("#meta");
const note = document.querySelector("#note");
const revision = document.querySelector("#revision");

let ready = false;
let selected = SCENARIOS[0].id;

policy.textContent = POLICY;

function select(id) {
  selected = id;
  const scenario = SCENARIOS.find((item) => item.id === id);
  ticket.value = scenario.ticket;
  for (const button of scenarios.querySelectorAll("button")) {
    button.setAttribute("aria-selected", button.dataset.id === id ? "true" : "false");
  }
  results.innerHTML = `<p class="empty">Pick a ticket and ask. Each bar is the probability of an option you defined. Nothing is generated outside that list.</p>`;
  meta.textContent = "Three typed questions, scored together.";
  note.textContent = "";
}

for (const scenario of SCENARIOS) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.id = scenario.id;
  button.textContent = scenario.label;
  button.setAttribute("role", "tab");
  button.addEventListener("click", () => select(scenario.id));
  scenarios.append(button);
}
select(selected);

function setStatus(state, text) {
  status.dataset.state = state;
  status.textContent = text;
}

function formatDuration(ms) {
  if (ms < 10000) return `${(ms / 1000).toFixed(1)} s`;
  const total = Math.round(ms / 1000);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return minutes ? `${minutes} min ${seconds} s` : `${seconds} s`;
}

function bars(pairs, winner) {
  return pairs
    .map(([name, probability]) => {
      const pct = Math.round(probability * 1000) / 10;
      const win = name === winner ? " winner" : "";
      return `<div class="row${win}"><span class="name">${escapeHtml(name)}</span><span class="track"><span class="bar" style="width:${Math.max(pct, 1.5)}%"></span></span><span class="pct">${pct.toFixed(1)}%</span></div>`;
    })
    .join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}

function render(payload) {
  const answers = payload.answers || {};
  const levels = QUESTIONS.urgency.criteria;
  const blocks = [];

  const department = answers.department;
  if (department?.probabilities) {
    const pairs = Object.entries(department.probabilities).sort((a, b) => b[1] - a[1]);
    blocks.push(`<article class="card"><h3>Team · ${escapeHtml(department.choice)}</h3><p class="prompt">${escapeHtml(QUESTIONS.department.instructions)}</p>${bars(pairs, department.choice)}</article>`);
  }

  const refund = answers.refund;
  if (refund && typeof refund.noul === "number") {
    const yes = refund.noul;
    blocks.push(`<article class="card"><h3>Refund asked · ${yes >= 0.5 ? "yes" : "no"}</h3><p class="prompt">${escapeHtml(QUESTIONS.refund.instructions)}</p>${bars([["yes", yes], ["no", 1 - yes]], yes >= 0.5 ? "yes" : "no")}</article>`);
  }

  const urgency = answers.urgency;
  if (urgency?.probabilities) {
    const pairs = Object.entries(urgency.probabilities)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([index, probability]) => [levels[Number(index)] || index, probability]);
    const chosen = pairs.reduce((best, pair) => (pair[1] > best[1] ? pair : best))[0];
    const expected = typeof urgency.score === "number" ? ` · expected ${urgency.score.toFixed(2)}` : "";
    blocks.push(`<article class="card"><h3>Urgency · ${escapeHtml(chosen)}${expected}</h3><p class="prompt">${escapeHtml(QUESTIONS.urgency.instructions)}</p>${bars(pairs, chosen)}</article>`);
  }

  results.innerHTML = blocks.join("") || `<p class="empty">${escapeHtml(JSON.stringify(answers))}</p>`;
  const tokens = payload.usage?.input_tokens;
  meta.textContent = formatDuration(payload.request_ms) + (tokens != null ? ` · ${tokens} input tokens` : "");
}

async function refresh() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    revision.textContent = (health.revision || "").slice(0, 12);
    if (health.status === "ready") {
      ready = true;
      ask.disabled = false;
      setStatus("ready", `Ready · ${health.device || "cpu"} · ${health.dtype || "bf16"}`);
      return;
    }
    ready = false;
    ask.disabled = true;
    if (health.status === "error") {
      setStatus("error", "Could not load the model");
      note.textContent = health.error || "";
      return;
    }
    setStatus("loading", "Loading weights…");
  } catch {
    setStatus("loading", "Waiting for the demo server…");
  }
}

ask.addEventListener("click", async () => {
  if (!ready) return;
  ask.disabled = true;
  note.textContent = "Scoring on CPU. On this laptop a pass can take a few minutes.";
  meta.textContent = "Scoring…";
  try {
    const response = await fetch("/api/systemone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state: { ticket: ticket.value.trim(), refund_policy: POLICY },
        questions: QUESTIONS,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    note.textContent = "";
    render(payload);
  } catch (error) {
    note.textContent = error.message;
  } finally {
    ask.disabled = !ready;
  }
});

refresh();
setInterval(refresh, 2000);
