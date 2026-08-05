import { expect, test } from "@playwright/test";

test("searches through the UI and renders grounded sources", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: /Ask the calendar/i })).toBeVisible();
  await page.getByLabel("Your question").fill("What does antirequisite mean?");
  await page.getByRole("button", { name: "Search", exact: true }).click();

  await expect(page.getByText("Grounded answer")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ranked sources" })).toBeVisible();
  await expect(page.locator(".sources article")).toHaveCount(5);
  await expect(page.getByText(/Synthetic demo content only/i)).toBeVisible();
});

test("runs every retrieval mode and validates short input", async ({ page }) => {
  await page.goto("/");
  const searchButton = page.getByRole("button", { name: "Search", exact: true });
  for (const mode of ["bm25", "dense", "hybrid", "reranked"]) {
    await page.getByLabel(mode).check();
    await page.getByLabel("Your question").fill("What is a course waitlist?");
    const [response] = await Promise.all([
      page.waitForResponse((candidate) => candidate.url().endsWith("/answer")),
      searchButton.click(),
    ]);
    expect(response.ok()).toBeTruthy();
    await expect(page.locator(".sources article")).toHaveCount(5);
  }

  await page.getByLabel("Your question").fill("x");
  await searchButton.click();
  await expect(page.locator(".form-error")).toContainText("at least three characters");
});
