# Load the data from the CSV file
my_data <- read.csv("archive/voice.csv")

# Perform different types of analysis on the data and save their graphs
library(dplyr)
library(ggplot2)
library(cluster)
library(tidyverse)
library(gridExtra)


if (!dir.exists("R_Graphs")) {
  dir.create("R_Graphs")
}

# PCA 
# Perform PCA
pca_result <- prcomp(my_data[,1:20], scale = TRUE)

# Extract the principal components
pc1 <- pca_result$x[,1]
pc2 <- pca_result$x[,2]

# Create a data frame with the principal components and labels
pca_df <- data.frame(pc1, pc2, label = my_data$label)

# Create a scatter plot of the results
ggplot(pca_df, aes(x = pc1, y = pc2, color = label)) +
  geom_point() +
  labs(title = "PCA Results", x = "Principal Component 1", y = "Principal Component 2")
ggsave("R_Graphs/pca_results.png")



# Load the data from the CSV file
my_data <- read.csv("archive/voice.csv")

col_names <- names(my_data)
# Create a list of plots for each label
plot_list <- list()
for (col_name in col_names) {
  if (col_name != "label") {
    data <- my_data[, c("label", col_name)]
    p <- ggplot(data, aes(x = label, y = .data[[col_name]], fill = label)) +
      geom_boxplot() +
      labs(title = NULL, x = NULL, y = col_name) +
      theme_classic()
    plot_list[[col_name]] <- p
    ggsave(sprintf("R_Graphs/%s.png", col_name))
  }
}

# Arrange the plots in a grid
grid_arranged <- grid.arrange(grobs = plot_list, ncol = 5)

# Save the grid as a PNG file
ggsave("grid_arranged.png", plot = grid_arranged)


