const PROTOTYPE_NOTICE =
  "Demo UI is connected to the FastAPI backend. Business logic runs on the server.";

document.documentElement.dataset.demoStage = "fastapi-connected";
console.info(PROTOTYPE_NOTICE);

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const revealItems = document.querySelectorAll(".reveal, .decision-line");

if (prefersReducedMotion) {
  revealItems.forEach((item) => item.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.16 }
  );

  revealItems.forEach((item) => revealObserver.observe(item));
}

const factsStory = document.querySelector('[data-story="facts"]');
const decisionStory = document.querySelector('[data-story="decision"]');

const factTitles = [
  "Сначала был текст.",
  "Потом проявился смысл.",
  "Детали обрели значение.",
  "И текст стал структурой."
];

const factTexts = [
  "Обычный проектный бриф: свободная форма, разная детализация, важные факты вперемешку с контекстом.",
  "Система выделяет цель проекта и фактический объём работы.",
  "Сроки, материалы и ограничения становятся отдельными сигналами для дальнейшей проверки.",
  "Свободный текст преобразован в структурированные факты, которые можно проверять дальше."
];

const decisionTitles = [
  "Система видит не только то, что уже известно.",
  "Она оценивает соответствие самой работы.",
  "Она видит, где проект становится рискованным.",
  "Теперь мы понимаем проект."
];

const decisionTexts = [
  "Она показывает, чего не хватает, и отделяет критичные пробелы от деталей, которые можно уточнить позже.",
  "Явные задачи из брифа сопоставляются с правилами направления и специализации.",
  "Факты превращаются в сигналы: большой объём, сжатый срок, ограничения первого релиза.",
  "Аналитический портрет готов: определённость, соответствие и риски собраны в одном месте."
];

const decisionLabels = [
  "COMPLETENESS · RULES",
  "TRAFFIC LIGHT · ASSESSMENT",
  "ASSESSMENT · AI",
  "STRUCTURED SIGNALS"
];

let ticking = false;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function storyProgress(section) {
  const rect = section.getBoundingClientRect();
  const scrollable = rect.height - window.innerHeight;
  if (scrollable <= 0) return 1;
  return clamp(-rect.top / scrollable, 0, 1);
}

function segmentState(progress, count) {
  const scaled = progress * count;
  const index = clamp(Math.floor(scaled), 0, count - 1);
  const local = clamp(scaled - index, 0, 1);
  return { index, local };
}

function setFactsStory(step, localProgress) {
  const title = factsStory.querySelector("[data-story-title]");
  const text = factsStory.querySelector("[data-story-text]");
  if (title) title.textContent = factTitles[step];
  if (text) text.textContent = factTexts[step];

  factsStory.querySelectorAll("mark").forEach((mark) => mark.classList.remove("active"));
  factsStory.querySelectorAll(".fact-row").forEach((row) => row.classList.remove("active"));

  const factsPanel = factsStory.querySelector(".facts-panel");
  if (factsPanel) {
    factsPanel.classList.toggle("has-facts", step > 0);
    factsPanel.classList.toggle("assembled", step === factTitles.length - 1);
    factsPanel.style.setProperty("--story-local-progress", localProgress.toFixed(3));
  }

  const activeByStep = [
    [],
    ["goal", "tasks"],
    ["goal", "tasks", "deadline", "materials", "constraints"],
    ["goal", "tasks", "deadline", "materials", "constraints"]
  ];

  activeByStep[step].forEach((key) => {
    factsStory.querySelectorAll(`[data-highlight="${key}"]`).forEach((item) => item.classList.add("active"));
    const row = factsStory.querySelector(`[data-fact="${key}"]`);
    if (row) {
      row.classList.add("active");
      row.classList.toggle("connected", step < factTitles.length - 1);
    }
  });
}

if (analyzeButton && analysisState && mockResult && briefTextarea) {
  analyzeButton.addEventListener("click", async () => {
    analyzeButton.disabled = true;
    analysisState.textContent = "Анализируем бриф…";
    mockResult.classList.add("hidden");

    try {
      const response = await window.fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ brief: briefTextarea.value })
      });
      const payload = await response.json().catch(() => null);

      if (!response.ok || payload?.ok === false) {
        throw new Error("Analyze request failed");
      }

      renderAnalysisResult(payload);
      analysisState.textContent = "Анализ готов.";
      mockResult.classList.remove("hidden");
    } catch (error) {
      analysisState.textContent =
        "Не удалось выполнить анализ. Попробуйте повторить запрос позже.";
    } finally {
      analyzeButton.disabled = false;
    }
  });
}

function setDecisionStory(step, localProgress) {
  const title = decisionStory.querySelector("[data-decision-title]");
  const text = decisionStory.querySelector("[data-decision-text]");
  const label = decisionStory.querySelector("[data-decision-label]");
  if (title) title.textContent = decisionTitles[step];
  if (text) text.textContent = decisionTexts[step];
  if (label) label.textContent = decisionLabels[step];

  const verifiedObject = decisionStory.querySelector(".verified-object");
  const traffic = decisionStory.querySelector('[data-layer="traffic"]');
  const risks = decisionStory.querySelector('[data-layer="risks"]');
  const portrait = decisionStory.querySelector('[data-layer="portrait"]');

  if (verifiedObject) {
    verifiedObject.classList.toggle("show-checks", step >= 0);
    verifiedObject.style.setProperty("--story-local-progress", localProgress.toFixed(3));
  }

  decisionStory.querySelectorAll(".verified-row").forEach((row) => {
    row.classList.remove("focus");
  });
  const focusMap = {
    1: ["tasks"],
    2: ["tasks", "deadline"],
    3: ["goal", "tasks", "deadline", "materials", "roles"]
  };
  (focusMap[step] || []).forEach((key) => {
    const row = decisionStory.querySelector(`[data-layer="${key}"]`);
    if (row) row.classList.add("focus");
  });

  if (traffic) traffic.classList.toggle("active", step >= 1);
  if (risks) risks.classList.toggle("active", step >= 2);
  if (portrait) portrait.classList.toggle("active", step >= 3);
}

function updateStories() {
  ticking = false;
  if (factsStory) {
    const state = segmentState(storyProgress(factsStory), factTitles.length);
    setFactsStory(state.index, state.local);
  }
  if (decisionStory) {
    const state = segmentState(storyProgress(decisionStory), decisionTitles.length);
    setDecisionStory(state.index, state.local);
  }
}

function requestStoryUpdate() {
  if (!ticking) {
    ticking = true;
    window.requestAnimationFrame(updateStories);
  }
}

if (prefersReducedMotion) {
  if (factsStory) setFactsStory(factTitles.length - 1, 1);
  if (decisionStory) setDecisionStory(decisionTitles.length - 1, 1);
} else {
  window.addEventListener("scroll", requestStoryUpdate, { passive: true });
  window.addEventListener("resize", requestStoryUpdate);
  updateStories();
}

const analyzeButton = document.querySelector(".analyze-button");
const analysisState = document.querySelector(".analysis-state");
const mockResult = document.querySelector(".mock-result");
const briefTextarea = document.querySelector(".demo-input textarea");

function textOrFallback(value, fallback) {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : fallback;
}

function numberOrFallback(value, fallback) {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : fallback;
}

function renderAnalysisResult(payload) {
  const decision = payload?.decision ?? {};
  const customerResponse = payload?.customer_response ?? {};
  const completeness = payload?.completeness ?? {};
  const trafficLight = payload?.traffic_light ?? {};
  const risks = payload?.risks ?? {};

  const finalStatus = textOrFallback(decision.final_status, "UNKNOWN");
  const summary = textOrFallback(decision.summary, "Результат анализа получен.");
  const responseText = textOrFallback(customerResponse.text, "Текст ответа заказчику пока отсутствует.");
  const completenessLevel = textOrFallback(completeness.level, "n/a");
  const trafficStatus = textOrFallback(trafficLight.status, "n/a");
  const riskCount = numberOrFallback(risks.count, "n/a");

  mockResult.innerHTML = "";

  const status = document.createElement("strong");
  status.textContent = finalStatus;

  const summaryParagraph = document.createElement("p");
  summaryParagraph.textContent = summary;

  const metrics = document.createElement("div");
  metrics.className = "demo-summary";
  [
    `Решение: ${finalStatus}`,
    `Полнота: ${completenessLevel}`,
    `Traffic Light: ${trafficStatus}`,
    `Риски: ${riskCount}`
  ].forEach((label) => {
    const item = document.createElement("span");
    item.textContent = label;
    metrics.appendChild(item);
  });

  const customerParagraph = document.createElement("p");
  customerParagraph.textContent = responseText;

  mockResult.append(status, summaryParagraph, metrics, customerParagraph);
}

if (false && analyzeButton && analysisState && mockResult) {
  analyzeButton.addEventListener("click", () => {
    analyzeButton.disabled = true;
    analysisState.textContent = "Анализируем бриф…";
    mockResult.classList.add("hidden");

    window.setTimeout(() => {
      analysisState.textContent = "Mock UX готов. Backend не вызывался.";
      mockResult.classList.remove("hidden");
      analyzeButton.disabled = false;
    }, 850);
  });
}

if (analyzeButton && analysisState && mockResult && briefTextarea) {
  analyzeButton.addEventListener("click", async () => {
    analyzeButton.disabled = true;
    analysisState.textContent = "Анализируем бриф…";
    mockResult.classList.add("hidden");

    try {
      const response = await window.fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ brief: briefTextarea.value })
      });
      const payload = await response.json().catch(() => null);

      if (!response.ok || payload?.ok === false) {
        throw new Error("Analyze request failed");
      }

      renderAnalysisResult(payload);
      analysisState.textContent = "Анализ готов.";
      mockResult.classList.remove("hidden");
    } catch (error) {
      analysisState.textContent =
        "Не удалось выполнить анализ. Попробуйте повторить запрос позже.";
    } finally {
      analyzeButton.disabled = false;
    }
  });
}
