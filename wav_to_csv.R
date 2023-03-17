
library(warbleR)

file_path <- "common_voice_en_33152279.mp3.wav"

dataframe <- data.frame(list = c("sound.files", "selec", "start", "end"))

dataframe <- data.frame(file_path, 2, 1, 2)

names(dataframe) <- c("sound.files", "selec", "start", "end")

a <- specan(X=dataframe, bp=c(0,1))

write.csv(a, "Testing Output\\specan.csv")
