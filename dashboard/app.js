"use strict";

const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "—").replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ],
  );
const number = (value) =>
  value !== null &&
  value !== undefined &&
  value !== "" &&
  Number.isFinite(Number(value))
    ? Number(value).toLocaleString()
    : "—";
const money = (value) =>
  Number.isFinite(Number(value)) && value !== null
    ? new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
      }).format(Number(value))
    : "—";
const percent = (value) =>
  Number.isFinite(Number(value)) && value !== null
    ? `${Math.round(Number(value) * 100)}%`
    : "—";
const time = (value) => {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed)
    ? new Date(parsed).toLocaleTimeString([], {
        hour: "numeric",
        minute: "2-digit",
      })
    : "—";
};
const spotName = (spot, rules) => rules?.spot_names?.[spot] || spot || "—";
const listText = (items) =>
  Array.isArray(items) && items.length
    ? items
        .map((item) =>
          typeof item === "string"
            ? item
            : item.text ||
              [item.name, item.spot].filter(Boolean).join(" at ") ||
              JSON.stringify(item),
        )
        .join(" · ")
    : "None";
let current = null;

function setNotice(message, kind = "") {
  const el = $("notice");
  el.textContent = message;
  el.className = `notice ${kind}`;
}

function renderStatus(data) {
  const state = data.state;
  const dot = $("refresh-dot");
  dot.className = "refresh-dot";
  if (!state) {
    $("refresh-label").textContent = data.log_exists
      ? "Waiting for state snapshot"
      : "No bot data yet";
    $("updated-at").textContent = "—";
    dot.classList.add("warn");
    setNotice(
      "No current state snapshot. If the bot is running, pull the latest code and restart it after its current order; the updated bot writes logs/latest-state.json.",
      "warn",
    );
    return;
  }
  const observed = Date.parse(state.observed_at);
  const phaseEnd = Date.parse(state.phase_ends_at);
  const serverAtCapture = Date.parse(state.server_time);
  const serverNow =
    Number.isFinite(serverAtCapture) && Number.isFinite(observed)
      ? Date.now() + serverAtCapture - observed
      : Date.now();
  const expired =
    Number.isFinite(phaseEnd) &&
    serverNow > phaseEnd + 60000 &&
    state.phase !== "over";
  const age = Number.isFinite(observed)
    ? Math.max(0, Math.round((Date.now() - observed) / 1000))
    : null;
  $("updated-at").textContent =
    age === null
      ? "Capture time unknown"
      : age < 60
        ? `${age}s since capture`
        : `${Math.floor(age / 60)}m since capture`;
  $("refresh-label").textContent = expired
    ? "Check bot terminal"
    : "Reading local bot data";
  dot.classList.add(expired ? "warn" : "good");
  if (expired)
    setNotice(
      "This snapshot is from a phase that should have ended. Check that the bot terminal is still running and connected.",
      "warn",
    );
  else if (state.phase === "over")
    setNotice(
      "This market is over. Results below are from its final saved state.",
      "good",
    );
  else setNotice("");
  $("phase").textContent = state.phase || "—";
  $("phase-detail").textContent = state.orders_open
    ? `Orders open for day ${state.order_for_day ?? "—"}`
    : state.phase === "night"
      ? "Orders closed"
      : "Waiting for next order window";
  $("phase-end").textContent = Number.isFinite(phaseEnd)
    ? `Ends ${time(state.phase_ends_at)}`
    : "No phase end published";
  $("progress").textContent =
    `${state.market ?? "—"} / ${state.markets ?? "—"} · ${state.day ?? "—"} / ${state.days ?? "—"}`;
  $("mode").textContent = `${state.mode || "—"} · market / completed day`;
  $("cash").textContent = money(state.you?.cash);
  $("state-details").textContent =
    `State version ${state.version ?? "—"} · Server time ${time(state.server_time)} · Snapshot saved ${time(state.observed_at)}`;
  const trucks = [...(state.trucks || [])].sort(
    (a, b) => Number(b.cash || 0) - Number(a.cash || 0),
  );
  const rank = trucks.findIndex((truck) => truck.id === state.you?.id);
  $("standing").textContent =
    rank >= 0
      ? `#${rank + 1} of ${trucks.length} trucks`
      : `${trucks.length} trucks in market`;
  updateCountdown();
}

function updateCountdown() {
  const state = current?.state;
  const end = Date.parse(state?.phase_ends_at);
  const observed = Date.parse(state?.observed_at);
  const server = Date.parse(state?.server_time);
  if (
    !Number.isFinite(end) ||
    !Number.isFinite(observed) ||
    !Number.isFinite(server)
  ) {
    $("countdown").textContent = "—";
    return;
  }
  const remaining = Math.max(
    0,
    Math.ceil((end - (Date.now() + server - observed)) / 1000),
  );
  $("countdown").textContent =
    remaining === 0
      ? "00:00"
      : `${String(Math.floor(remaining / 60)).padStart(2, "0")}:${String(remaining % 60).padStart(2, "0")}`;
}

function renderOrder(data) {
  const state = data.state || {},
    rules = state.rules || {};
  const submitted = data.latest_order;
  let order = state.you?.order;
  let orderDay = state.order_for_day;
  if (
    !order &&
    submitted &&
    (submitted.day === orderDay ||
      (state.phase === "day" && submitted.day === state.day))
  ) {
    order = submitted.order;
    orderDay = submitted.day;
  }
  const badge = $("order-badge");
  if (!order) {
    badge.textContent = state.orders_open ? "Needs order" : "No next order";
    badge.className = `pill ${state.orders_open ? "warn" : ""}`;
    $("order").className = "order-content empty-copy";
    $("order").textContent = state.orders_open
      ? "The order window is open, but no order appears in the saved state yet. Check the bot terminal."
      : "No order is queued right now. The bot will decide when the next night opens.";
    return;
  }
  badge.textContent =
    state.orders_open || state.phase === "night"
      ? "Submitted"
      : "Latest submitted";
  badge.className = "pill good";
  const costPerMeal = Number(rules.costs?.[order.quality]);
  const foodCost = Number.isFinite(costPerMeal)
    ? costPerMeal * Number(order.meals || 0)
    : null;
  const breakEven =
    Number(order.price) > 0 && foodCost !== null
      ? Math.ceil(foodCost / Number(order.price))
      : null;
  $("order").className = "order-content";
  $("order").innerHTML =
    `<div class="order-location">${esc(spotName(order.spot, rules))}<small>Day ${esc(orderDay ?? "—")} · ${esc(order.quality || "—")} quality</small></div><div class="order-stat"><span class="label">Price / meal</span><strong>${esc(money(order.price))}</strong></div><div class="order-stat"><span class="label">Meals cooked</span><strong>${esc(number(order.meals))}</strong></div><div class="order-stat"><span class="label">Food cost</span><strong>${esc(money(foodCost))}</strong></div><div class="order-note">${breakEven === null ? "Break-even unavailable" : `Needs ${esc(number(breakEven))} sales to cover food cost`} · Budget ${esc(money(rules.budget))} · Unsold meals spoil</div>`;
}

function renderAgent(data) {
  const plan = data.plan,
    state = data.state || {},
    rules = state.rules || {};
  const badge = $("agent-badge"),
    summary = $("agent-summary"),
    detail = $("agent-detail");
  const open = state.order_for_day != null;
  const stale =
    plan &&
    open &&
    (plan.day !== state.order_for_day ||
      plan.market !== state.market ||
      plan.mode !== state.mode);
  if (!plan || !plan.order || stale) {
    badge.textContent = plan ? "Stale" : "No plan";
    badge.className = "pill warn";
    summary.className = "empty-copy";
    summary.textContent = plan
      ? `The saved plan is for day ${plan.day ?? "—"}, but the coming day is ${state.order_for_day}. The agent has not planned this day yet, so these numbers are out of date.`
      : "No agent plan saved yet. Restart the bot with the latest code; it writes logs/agent-plan.json when it plans an order.";
    detail.hidden = true;
    return;
  }
  badge.textContent = open ? `Day ${plan.day} forecast` : `Last plan · day ${plan.day}`;
  badge.className = "pill";
  summary.className = "agent-summary";
  const kpis = [
    ["Expected profit", money(plan.expected_profit)],
    ["Expected sold", `${number(plan.expected_sold)} of ${number(plan.order.meals)}`],
    ["Chance of loss", percent(plan.p_loss)],
    [
      "Risk weight",
      `${number(plan.risk_weight)} · ${number(plan.losing_streak)} losing in a row`,
    ],
  ];
  summary.innerHTML = `<div class="agent-kpis">${kpis.map(([label, value]) => `<div class="last-kpi"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("")}</div><p class="subtle-line">${open ? "" : "No order window is open; this is the last plan. "}Forecast for day ${esc(plan.day)}, not a result. Forecast bias ×${esc(plan.forecast_bias)}, noise ±${esc(percent(plan.forecast_noise))}. Planned in ${esc(plan.elapsed_ms)} ms.</p>`;
  detail.hidden = false;
  const chosen = plan.order?.spot;
  $("agent-rows").innerHTML = (plan.alternatives || [])
    .map(
      (row) =>
        `<tr class="${row.spot === chosen ? "selected" : ""}"><td>${esc(spotName(row.spot, rules))}</td><td>${esc(row.quality)} · ${esc(money(row.price))} × ${esc(number(row.meals))}</td><td>${esc(money(row.expected_profit))}</td><td>${esc(percent(row.p_loss))}</td><td>${esc(plan.rivals?.[row.spot]?.expected ?? "—")}</td></tr>`,
    )
    .join("") ||
    '<tr><td colspan="5" class="empty-copy">No alternatives saved.</td></tr>';
  const profiles = [...(plan.rival_profiles || [])].sort(
    (a, b) => Number(b.chance || 0) - Number(a.chance || 0),
  );
  $("agent-rivals").innerHTML = profiles.length
    ? profiles
        .map(
          (rival) =>
            `<div class="agent-rival"><strong>${esc(rival.name)}</strong><span>${rival.most_likely ? `${esc(spotName(rival.most_likely, rules))} ${esc(percent(rival.chance))}` : "—"}</span><span>${esc(rival.quality)} · ${esc(money(rival.price))}</span></div>`,
        )
        .join("")
    : '<p class="empty-copy">No rival predictions saved.</p>';
  const checks = [...(plan.checks || [])].reverse();
  $("agent-checks").innerHTML = checks.length
    ? checks
        .map(
          (row) =>
            `<tr><td>${esc(row.day)}</td><td>${esc(spotName(row.spot, rules))}</td><td>${esc(row.expected_sold == null ? "—" : Math.round(row.expected_sold))} / ${esc(number(row.actual_sold))}</td><td>${esc(money(row.expected_profit))} / ${esc(money(row.actual_profit))}</td><td>${esc(row.expected_rivals ?? "—")} / ${esc(row.actual_rivals ?? "—")}</td></tr>`,
        )
        .join("")
    : '<tr><td colspan="5" class="empty-copy">No checked days yet. The first one appears after a planned day finishes.</td></tr>';
}

function renderForecast(state) {
  const forecast = state?.forecast || {},
    rules = state?.rules || {},
    last = state?.last_day || {};
  $("forecast-day").textContent =
    forecast.day === undefined ? "—" : `DAY ${forecast.day}`;
  const weather =
    forecast.weather &&
    Object.entries(forecast.weather)
      .map(([name, chance]) => `${esc(name)} ${esc(percent(chance))}`)
      .join(" / ");
  $("forecast-meta").innerHTML =
    forecast.day === undefined
      ? "Waiting for forecast…"
      : `<span><strong>Weather chances</strong> ${weather || "—"}</span><span><strong>Events</strong> ${esc(listText(forecast.events))}</span>`;
  const spots = rules.spots || Object.keys(forecast.expected_customers || {});
  $("forecast-rows").innerHTML = spots.length
    ? spots
        .map((spot) => {
          const rivals = (last.spots?.[spot]?.trucks || []).filter(
            (truck) => truck.id !== state.you?.id,
          ).length;
          return `<tr class="${state.you?.order?.spot === spot ? "selected" : ""}"><td>${esc(spotName(spot, rules))}</td><td>${esc(number(forecast.expected_customers?.[spot]))}</td><td>${esc(forecast.crowds?.[spot] || "—")}</td><td>${esc(number(rivals))}</td><td>${esc(number(state.you?.reputation?.[spot]))}</td></tr>`;
        })
        .join("")
    : '<tr><td colspan="5" class="empty-copy">No forecast yet.</td></tr>';
}

function renderLastDay(state) {
  const last = state?.last_day || {},
    own = last.you || {},
    rules = state?.rules || {};
  $("last-day").textContent = last.day == null ? "—" : `DAY ${last.day}`;
  if (last.day == null) {
    $("last-summary").className = "last-summary empty-copy";
    $("last-summary").textContent = "Results will appear after a day ends.";
  } else {
    const sold = Number(own.sold || 0),
      meals = Number(own.meals || 0);
    const vals = [
      [
        "Your profit",
        money(own.profit),
        Number(own.profit || 0) < 0 ? "negative" : "positive",
      ],
      ["Meals sold", own.spot ? `${sold} / ${meals}` : "No order", ""],
      ["Spoiled", own.spot ? number(Math.max(0, meals - sold)) : "—", ""],
      ["Sell-through", meals ? percent(sold / meals) : "—", ""],
    ];
    $("last-summary").className = "last-summary";
    $("last-summary").innerHTML = vals
      .map(
        ([label, value, className]) =>
          `<div class="last-kpi"><span>${esc(label)}</span><strong class="${className}">${esc(value)}</strong></div>`,
      )
      .join("");
  }
  $("last-meta").textContent =
    last.day == null
      ? ""
      : `${own.spot ? `You parked at ${spotName(own.spot, rules)}. ` : "You had no order. "}Actual weather: ${last.weather || "unknown"}. Events: ${listText(last.events)}.`;
  const spots = rules.spots || Object.keys(last.spots || {});
  $("last-rows").innerHTML =
    spots.length && last.day != null
      ? spots
          .map((spot) => {
            const actual = last.spots?.[spot] || {};
            return `<tr class="${own.spot === spot ? "selected" : ""}"><td>${esc(spotName(spot, rules))}</td><td>${esc(number(actual.customers))}</td><td>${esc(number(actual.hungry))}</td><td>${esc(number((actual.trucks || []).length))}</td></tr>`;
          })
          .join("")
      : '<tr><td colspan="4" class="empty-copy">No day results yet.</td></tr>';
  $("rivals").innerHTML =
    last.day == null
      ? '<p class="empty-copy">No rivals to inspect yet.</p>'
      : spots
          .map((spot) => {
            const trucks = last.spots?.[spot]?.trucks || [];
            return `<div class="rival-spot"><h3>${esc(spotName(spot, rules))}</h3>${trucks.length ? trucks.map((truck) => `<p class="${truck.id === state.you?.id ? "mine" : ""}">${esc(truck.truck_name || "Truck")} · ${esc(truck.quality || "—")} · ${esc(money(truck.price))} · sold ${esc(number(truck.sold))} / ${esc(number(truck.meals))}</p>`).join("") : "<p>No trucks here.</p>"}</div>`;
          })
          .join("");
}

function renderPerformance(data) {
  const s = data.summary,
    rows = data.results || [];
  if (!s.days) {
    $("performance").className = "empty-copy";
    $("performance").textContent =
      "No completed days logged for this market yet. The first result appears after a day ends.";
    return;
  }
  $("performance").className = "";
  const metrics = [
    ["Completed days", number(s.days)],
    ["Average / day", money(s.average_profit)],
    ["Sold / cooked", `${number(s.sold)} / ${number(s.cooked)}`],
    ["Sell-through", percent(s.sell_through)],
    ["Spoiled meals", number(s.spoiled)],
    ["Stockouts", number(s.stockouts)],
    ["Missed days", number(s.missed)],
    ["API errors", number(s.errors)],
    [
      "Median order time",
      s.median_order_ms == null ? "—" : `${number(s.median_order_ms)} ms`,
    ],
  ];
  const recent = rows.slice(-5).reverse();
  $("performance").innerHTML =
    `<div class="metric-total"><span>Total profit</span><strong>${esc(money(s.profit))}</strong></div><div class="metric-grid">${metrics.map(([label, value]) => `<div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("")}</div><div class="subtle-line">Recent days</div><div class="table-scroll"><table><thead><tr><th>Day</th><th>Spot</th><th>Sold</th><th>Profit</th></tr></thead><tbody>${recent.map((row) => `<tr><td>${esc(row.day)}</td><td>${esc(spotName(row.spot, data.state?.rules))}</td><td>${esc(number(row.sold))}/${esc(number(row.meals))}</td><td>${esc(money(row.profit))}</td></tr>`).join("")}</tbody></table></div>${s.latest_error ? `<p class="subtle-line">Latest API error: ${esc(s.latest_error.error_type || "error")}${s.latest_error.status ? ` ${esc(s.latest_error.status)}` : ""} at ${esc(time(s.latest_error.at))}</p>` : ""}`;
}

function renderRules(state) {
  const rules = state?.rules || {};
  if (!Object.keys(rules).length) {
    $("rules").textContent = "Waiting for game rules…";
    return;
  }
  const rows = [
    ["Food budget / day", money(rules.budget)],
    ["Price range", `${money(rules.price_min)}–${money(rules.price_max)}`],
    ...Object.entries(rules.costs || {}).map(([quality, cost]) => [
      `${quality} cost / meal`,
      money(cost),
    ]),
    ...Object.entries(rules.values || {}).map(([quality, value]) => [
      `${quality} value`,
      number(value),
    ]),
  ];
  $("rules").innerHTML = rows
    .map(
      ([label, value]) =>
        `<div class="rule-row"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`,
    )
    .join("");
}

function renderStandings(state) {
  const trucks = [...(state?.trucks || [])].sort(
    (a, b) => Number(b.cash || 0) - Number(a.cash || 0),
  );
  $("standings-list").innerHTML = trucks.length
    ? trucks
        .slice(0, 12)
        .map(
          (truck, index) =>
            `<li class="${truck.id === state.you?.id ? "mine" : ""}"><span class="rank">${index + 1}</span><span class="name" title="${esc(truck.company_name || "")}">${esc(truck.truck_name || "Truck")}</span><span class="amount">${esc(money(truck.cash))}</span></li>`,
        )
        .join("")
    : '<li class="empty-copy">No trucks listed yet.</li>';
}

function renderAnnouncements(state) {
  const news = [...(state?.announcements || [])].reverse().slice(0, 5);
  $("announcements").innerHTML = news.length
    ? news
        .map(
          (item) =>
            `<div class="news-item">${esc(typeof item === "string" ? item : item.text || "Announcement")}<span>${esc(typeof item === "string" ? "" : time(item.at))}</span></div>`,
        )
        .join("")
    : "No announcements yet.";
}

function render(data) {
  current = data;
  renderStatus(data);
  renderOrder(data);
  renderAgent(data);
  renderForecast(data.state);
  renderLastDay(data.state);
  renderPerformance(data);
  renderRules(data.state);
  renderStandings(data.state);
  renderAnnouncements(data.state);
}

async function refresh() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("refresh-dot").className = "refresh-dot bad";
    $("refresh-label").textContent = "Dashboard cannot read local data";
    setNotice(
      `The dashboard could not refresh (${error.message}). Check that python3 dashboard.py is still running.`,
      "bad",
    );
  }
}

refresh();
setInterval(refresh, 5000);
setInterval(updateCountdown, 1000);
