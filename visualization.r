library(R4RNA)


args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Podaj ścieżkę do folderu jako argument!")
}
base_folder <- args[1]

helix_folder <- file.path("/results/helix_outputs")
amt_folder   <- file.path(base_folder, "amt")
output_folder <- "/results/visualizations"

dir.create(output_folder, showWarnings = FALSE)
old_files <- list.files(output_folder, full.names = TRUE)
if (length(old_files) > 0) {
  file.remove(old_files)
}

helix_files <- list.files(helix_folder, pattern = "\\.helix$", full.names = TRUE)

for (file_path in helix_files) {

  helix_data <- read.table(file_path, sep = ",", header = FALSE, stringsAsFactors = FALSE)
  colnames(helix_data) <- c("start", "end", "category", "length")
  
  helix_data$start  <- as.numeric(helix_data$start) + 1
  helix_data$end    <- as.numeric(helix_data$end) + 1
  helix_data$length <- as.numeric(helix_data$length)
  
  category_mapping <- c(
    "PredictedGoodNonCanonical" = 1,
    "PredictedBadNonCanonical"  = 2,
    "NotPredictedNonCanonical"  = 3
  )
  unknown_categories <- setdiff(unique(helix_data$category), names(category_mapping))
  if (length(unknown_categories) > 0) {
    stop("Unknown categories in .helix file: ", paste(unknown_categories, collapse = ", "))
  }
  helix_data$id <- category_mapping[helix_data$category]
  
  helix_df <- helix_data[, c("start", "end", "length", "id")]
  seq_length_helix <- max(helix_df$start, helix_df$end)
  attr(helix_df, "length") <- seq_length_helix
  
  helix_pred <- as.helix(helix_df)
  
  helix_data$category <- factor(
    helix_data$category,
    levels = c("PredictedGoodNonCanonical", "PredictedBadNonCanonical", "NotPredictedNonCanonical")
  )
  color_palette <- c("#107F80", "#FF0066", "#66CCFE")
  helix_pred$col <- color_palette[as.numeric(helix_data$category)]
  
  base_name <- sub("\\.helix$", "", basename(file_path))  
  amt_file_path <- file.path(amt_folder, base_name)
  
  if (!file.exists(amt_file_path)) {
    warning(paste("Missing .amt file for:", file_path))
    next
  }
  
  amt_matrix <- as.matrix(read.table(amt_file_path, sep = ",", header = FALSE))
  n_row_amt  <- nrow(amt_matrix)
  
  # canonical == 1, non-canonical > 1
  can_edges    <- which(amt_matrix == 1, arr.ind = TRUE)
  noncan_edges <- which(amt_matrix > 1,  arr.ind = TRUE)
  
  if (nrow(can_edges) > 0) {
    can_df <- data.frame(start = can_edges[, "row"], end = can_edges[, "col"], length = 1, id = 1)
  } else {
    can_df <- data.frame(start = numeric(0), end = numeric(0), length = numeric(0), id = numeric(0))
  }
  if (nrow(noncan_edges) > 0) {
    noncan_df <- data.frame(start = noncan_edges[, "row"], end = noncan_edges[, "col"], length = 1, id = 2)
  } else {
    noncan_df <- data.frame(start = numeric(0), end = numeric(0), length = numeric(0), id = numeric(0))
  }
  amt_df <- rbind(can_df, noncan_df)
  if (nrow(amt_df) == 0) {
    warning("Brak krawędzi w pliku .amt: ", amt_file_path)
    next
  }
  
  seq_length_amt <- max(amt_df$start, amt_df$end, n_row_amt)
  attr(amt_df, "length") <- seq_length_amt
  
  helix_amt <- as.helix(amt_df)
  helix_amt$id <- amt_df$id
  
  helix_amt$col <- ifelse(helix_amt$id == 1, "grey", "black")
  
  max_len <- max(seq_length_helix, seq_length_amt)
  attr(helix_pred, "length") <- max_len
  attr(helix_amt,  "length") <- max_len
  
  output_file <- file.path(output_folder, paste0(base_name, ".png"))
  png(output_file, width = 1000, height = 800)
  
  plotDoubleHelix(helix_pred, helix_amt, line = TRUE, arrow = TRUE, scale = FALSE, lwd = 2)
  
  dev.off()
}

cat("All visualizations saved in the folder:", output_folder, "\n")