(() => {
  const STORAGE_KEY = "dashboard-theme";

  function getPreferredTheme() {
    const saved = localStorage.getItem(STORAGE_KEY);

    if (saved === "dark" || saved === "light") {
      return saved;
    }

    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function updateIcon(button, theme) {
    if (!button) return;

    button.innerHTML =
      theme === "dark"
        ? "☀"
        : "☾";

    button.setAttribute(
      "aria-label",
      theme === "dark"
        ? "Cambiar a modo claro"
        : "Cambiar a modo oscuro"
    );
  }

  function applyTheme(theme, persist = false) {
    const dark = theme === "dark";

    document.body.classList.toggle("dark-mode", dark);

    if (persist) {
      localStorage.setItem(STORAGE_KEY, theme);
    }

    const button = document.getElementById("theme-toggle");
    updateIcon(button, theme);
  }

  function createToggle() {
    if (document.getElementById("theme-toggle")) return;

    const button = document.createElement("button");

    button.id = "theme-toggle";
    button.type = "button";

    button.addEventListener("click", () => {
      const nextTheme =
        document.body.classList.contains("dark-mode")
          ? "light"
          : "dark";

      applyTheme(nextTheme, true);
    });

    const navbar =
      document.querySelector(".navbar .container-fluid") ||
      document.querySelector(".navbar .container") ||
      document.querySelector(".navbar");

    if (navbar) {
      navbar.appendChild(button);
    } else {
      button.style.position = "fixed";
      button.style.top = "1rem";
      button.style.right = "1rem";
      button.style.zIndex = "9999";
      document.body.appendChild(button);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    createToggle();

    const initialTheme = getPreferredTheme();
    applyTheme(initialTheme);
  });
})();
