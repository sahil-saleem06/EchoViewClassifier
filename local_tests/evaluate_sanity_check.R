library(tidyverse)
library(yardstick)

# ── Load results ──────────────────────────────────────────────────────────────

results <- read_csv("local_tests/avi_labels.csv")

# All EchoNet-Dynamic videos are confirmed A4C — add ground truth column
results <- results |>
  mutate(
    truth     = factor("A4C", levels = c("A4C", "A2C", "PLAX", "PSAX", "OTHER")),
    predicted = factor(predicted_view, levels = c("A4C", "A2C", "PLAX", "PSAX", "OTHER"))
  )

# ── Metrics ───────────────────────────────────────────────────────────────────

cat("── Confusion Matrix ─────────────────────────────────────────────\n")
print(conf_mat(results, truth = truth, estimate = predicted))

cat("\n── Per-class Precision, Recall, F1 ─────────────────────────────\n")
metrics <- metric_set(precision, recall, f_meas)
print(metrics(results, truth = truth, estimate = predicted, estimator = "macro"))

cat("\n── Overall Accuracy ─────────────────────────────────────────────\n")
print(accuracy(results, truth = truth, estimate = predicted))
