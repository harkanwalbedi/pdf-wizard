// PDF Wizard: small progressive enhancements for the server-rendered pages.
//
// Loaded from 'self' only (CSP script-src 'self'); no inline scripts or event
// handler attributes. Data comes from quoted data-* attributes, and the DOM is
// built only with createElement, textContent, and replaceChildren. Nothing is
// written to the console and no native dialogs are used.
(() => {
  "use strict";

  const POLL_INTERVAL_MS = 1500;
  const POLL_BACKOFF_MS = 5000;
  const POLL_LIMIT_MS = 10 * 60 * 1000;

  const onReady = (fn) => {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
    } else {
      fn();
    }
  };

  const formatBytes = (bytes) => {
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${bytes} bytes`;
  };

  // Forms: disable the submit button once submitted, so a slow upload can't be
  // sent twice. The browser only fires "submit" after its own validation passes.
  const idleContent = new WeakMap();

  const initSubmitOnce = () => {
    document.querySelectorAll("form[data-submit-once]").forEach((form) => {
      form.addEventListener("submit", (event) => {
        if (form.dataset.submitted === "true") {
          event.preventDefault();
          return;
        }
        form.dataset.submitted = "true";
        form.querySelectorAll("button[type='submit']").forEach((button) => {
          button.disabled = true;
          button.setAttribute("aria-busy", "true");
          const busyLabel = button.dataset.busyLabel;
          if (busyLabel) {
            idleContent.set(button, Array.from(button.childNodes));
            button.replaceChildren(document.createTextNode(busyLabel));
          }
        });
      });
    });
  };

  // Pages restored from the back/forward cache keep their disabled buttons.
  const resetSubmittedForms = () => {
    document.querySelectorAll("form[data-submit-once]").forEach((form) => {
      delete form.dataset.submitted;
      form.querySelectorAll("button[type='submit']").forEach((button) => {
        button.disabled = false;
        button.removeAttribute("aria-busy");
        const original = idleContent.get(button);
        if (original) {
          button.replaceChildren(...original);
          idleContent.delete(button);
        }
      });
    });
  };

  // File inputs: list the chosen files and flag any over the size limit. The
  // server checks everything again; this is only a hint.
  const initFileLists = () => {
    document.querySelectorAll("input[type='file'][data-file-list]").forEach((input) => {
      const list = document.getElementById(input.dataset.fileList || "");
      if (!list) return;
      const maxBytes = Number(input.dataset.maxBytes) || 0;
      const render = () => {
        const items = Array.from(input.files || []).map((file) => {
          const item = document.createElement("li");
          const name = document.createElement("span");
          name.className = "file-name";
          name.textContent = file.name;
          name.title = file.name;
          const size = document.createElement("span");
          size.className = "file-size";
          size.textContent = formatBytes(file.size);
          item.append(name, size);
          if (maxBytes && file.size > maxBytes) {
            const warning = document.createElement("span");
            warning.className = "file-warning";
            warning.textContent = "Too large";
            item.append(warning);
          }
          return item;
        });
        list.replaceChildren(...items);
      };
      input.addEventListener("change", render);
      render();
    });
  };

  // Note: live character count.
  const initCounters = () => {
    document.querySelectorAll("textarea[data-counter]").forEach((area) => {
      const output = document.getElementById(area.dataset.counter || "");
      const max = Number(area.getAttribute("maxlength")) || 0;
      if (!output || !max) return;
      const update = () => {
        output.textContent = `${area.value.length.toLocaleString()} / ${max.toLocaleString()}`;
      };
      area.addEventListener("input", update);
      update();
    });
  };

  // Job page: poll the status API while the job is queued or running, then
  // reload so the server renders the results (or the error).
  const initStatusPolling = () => {
    const box = document.querySelector("[data-status-url]");
    if (!box) return;
    const url = box.dataset.statusUrl || "";
    if (!url.startsWith("/jobs/")) return;
    const text = document.getElementById("job-status-text");
    const labels = {
      queued: box.dataset.labelQueued || "",
      running: box.dataset.labelRunning || "",
    };
    const started = Date.now();

    const setText = (value) => {
      if (text && value && text.textContent !== value) text.textContent = value;
    };

    const poll = async () => {
      if (Date.now() - started > POLL_LIMIT_MS) {
        setText(box.dataset.timeoutText || "");
        return;
      }
      let delay = POLL_INTERVAL_MS;
      try {
        const response = await fetch(url, {
          method: "GET",
          credentials: "same-origin",
          cache: "no-store",
          headers: { Accept: "application/json" },
        });
        if (response.status === 401 || response.status === 404) {
          window.location.reload(); // signed out, or the job is gone: let the server decide
          return;
        }
        if (response.ok) {
          const data = await response.json();
          const status = data && typeof data.status === "string" ? data.status : "";
          if (status === "done" || status === "failed") {
            window.location.reload();
            return;
          }
          setText(labels[status] || "");
        } else {
          delay = POLL_BACKOFF_MS;
        }
      } catch {
        delay = POLL_BACKOFF_MS; // network hiccup: try again a bit later
      }
      window.setTimeout(poll, delay);
    };

    window.setTimeout(poll, POLL_INTERVAL_MS);
  };

  // Results table: status filter chips.
  const initFilters = () => {
    const bar = document.querySelector("[data-filter-bar]");
    if (!bar) return;
    const table = document.getElementById(bar.dataset.filterTarget || "");
    if (!table) return;
    const rows = Array.from(table.querySelectorAll("tbody tr[data-status]"));
    const buttons = Array.from(bar.querySelectorAll("button[data-filter]"));
    const empty = document.getElementById(bar.dataset.filterEmpty || "");
    if (!rows.length || !buttons.length) return;

    const apply = (filter) => {
      let shown = 0;
      rows.forEach((row) => {
        const match = filter === "all" || row.dataset.status === filter;
        row.hidden = !match;
        if (match) shown += 1;
      });
      buttons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.filter === filter));
      });
      if (empty) empty.hidden = shown !== 0;
    };

    buttons.forEach((button) => {
      button.addEventListener("click", () => apply(button.dataset.filter || "all"));
    });
    bar.hidden = false;
    apply("all");
  };

  onReady(() => {
    initSubmitOnce();
    initFileLists();
    initCounters();
    initStatusPolling();
    initFilters();
  });

  window.addEventListener("pageshow", (event) => {
    if (event.persisted) resetSubmittedForms();
  });
})();
