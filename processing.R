# Load the data from the CSV file
my_data <- read.csv("archive/voice.csv")

# Perform different types of analysis on the data and save their graphs
library(dplyr)
library(ggplot2)
library(cluster)


if (!dir.exists("R_Graphs")) {
  dir.create("R_Graphs")
}

# means
means_by_label <- my_data %>%
  group_by(label) %>%
  summarize_all(mean)

# Reshape the data to long format
library(tidyr)
means_long <- means_by_label %>%
  gather(key = "feature", value = "mean", -label)

# Create a bar plot of the means by label for each feature
ggplot(means_long, aes(x = label, y = mean, fill = feature)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Means by Label for Each Feature", x = "Label", y = "Mean") +
  facet_wrap(~ feature, scales = "free") # facet the plot by feature with free y-axis scales
ggsave("R_Graphs/means_by_label_each_feature.png")


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
