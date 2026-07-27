const { test, expect, chromium } = require("@playwright/test");

const consoleUrl = process.env.FW_CONSOLE_URL;
const token = process.env.FW_CONSOLE_TOKEN;
const screenshot = process.env.FW_CONSOLE_SCREENSHOT;
const executablePath = process.env.FW_PLAYWRIGHT_EXECUTABLE;
const emptyConsoleUrl = process.env.FW_EMPTY_CONSOLE_URL;
const emptyScreenshot = process.env.FW_EMPTY_CONSOLE_SCREENSHOT;

async function launchBrowser() {
  return chromium.launch({
    headless: false,
    ...(executablePath ? { executablePath } : {}),
  });
}

async function authenticate(page, url) {
  await page.goto(`${url}/console`);
  await expect(page.getByRole("heading", { name: "Accéder à la console du pod" })).toBeVisible();
  await page.getByLabel("Jeton d’authentification").fill(token);
  await page.getByLabel("Identifiant du réviseur").fill("qa-operator");
  await page.getByRole("button", { name: "Ouvrir la console" }).click();
  await expect(page.getByText("Connecté", { exact: true })).toBeVisible();
}

test("authenticated operator can inspect an in-pod case", async () => {
  test.skip(!consoleUrl || !token, "FW_CONSOLE_URL and FW_CONSOLE_TOKEN are required");
  const browser = await launchBrowser();
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    await authenticate(page, consoleUrl);
    await page.getByRole("button", { name: /Journées A-à-Z/ }).click();
    await expect(page.getByAltText("Aperçu du cas synthétique sélectionné")).toBeVisible();
    await expect(page.getByText("fid-000000", { exact: true })).toBeVisible();
    await expect(page.getByText("Aucun export avant validation", { exact: true })).toBeVisible();
    if (screenshot) {
      await page.screenshot({ path: screenshot, fullPage: true });
    }
  } finally {
    await browser.close();
  }
});

test("empty production volume never invents interface cases", async () => {
  test.skip(!emptyConsoleUrl || !token, "FW_EMPTY_CONSOLE_URL and FW_CONSOLE_TOKEN are required");
  const browser = await launchBrowser();
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    await authenticate(page, emptyConsoleUrl);
    await expect(page.getByText("Aucun cas produit", { exact: true })).toBeVisible();
    await expect(page.getByText("Aucun événement enregistré.", { exact: true })).toBeVisible();
    await expect(page.getByAltText("Aperçu du cas synthétique sélectionné")).toBeHidden();
    await expect(page.locator(".category-button").first()).toContainText("0 / 4 096");
    if (emptyScreenshot) {
      await page.screenshot({ path: emptyScreenshot, fullPage: true });
    }
  } finally {
    await browser.close();
  }
});
