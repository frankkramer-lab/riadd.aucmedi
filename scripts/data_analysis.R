# Libraries
library("ggplot2")
library("data.table")
library(dplyr)

# Data Path
path_data <- "/home/mudomini/data/RIADD/Training_Set/RFMiD_Training_Labels.csv"

# Create plot directory
dir.create(file.path("plots"), showWarnings = FALSE)

# Load dataset
dt <- fread(path_dir)

# Compute disease risk association to disease labels
counts <- apply(dt[, 3:30], 1, sum)
dt_counts <- dt[, 1:2]
dt_counts[, counts:= counts]
dt_counts_heat <- with(dt_counts, table(counts, Disease_Risk))

# Plot disease label association to risk
plot_risklabel <- ggplot(data=as.data.frame(dt_counts_heat), aes(x=factor(counts), y=Freq, fill=Disease_Risk)) +
  geom_col() +
  scale_fill_brewer(palette="Dark2") +
  theme_light() + 
  xlab("Number of Disease Labels") +
  ylab("Number of Samples") +
  ggtitle("Association between Disease Label Frequency and Risk")
png(file.path("plots", "risk_label_association.png"), width=1000, height=800, res=150)
plot_risklabel
dev.off()  

# Label frequency
dt_labels <- melt(sapply(dt[, 3:30], sum))
dt_labels <- rbind(dt_labels, Normal=sum(dt[, 2]==0))
dt_labels <- tibble::rownames_to_column(dt_labels, "class")

# Print label frequency
dt_labels

# Plot label frequency
plot_labels <- ggplot(dt_labels, aes(x=class, y=value, fill=class)) +
  geom_bar(stat="identity", width=0.75, col="black") +
  scale_y_continuous(breaks=seq(0, 420, 20), limits=c(0, 410)) +
  theme_light() + 
  theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1)) + 
  xlab("") +
  ylab("Number of Annotations") +
  ggtitle("Disease Label Annotation Frequency")
png(file.path("plots", "label_freq.png"), width=1400, height=800, res=150)
plot_labels
dev.off()  

# Prepare heatmap
dt_heat <- melt(dt, id.vars="ID", variable.name="class", value.name="value")
dt_heat[value==0]$value <- 2
dt_heat[value==1]$value <- 0
dt_heat[value==2]$value <- 1

# Plot Heatmap of label annotation
plot_heat <- ggplot(dt_heat, aes(ID, class, fill=value)) + 
  geom_tile() +
  scale_x_discrete(breaks=NULL) +
  scale_fill_distiller(palette="Blues") +
  theme_light() + 
  theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1)) +    
  xlab("Samples") +
  ylab("") +
  ggtitle("Disease Label Distribution")
png(file.path("plots", "label_heat.png"), width=1600, height=800, res=150)
plot_heat
dev.off()  
