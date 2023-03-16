library(tuneR)
library(seewave)

# Set the directory path where the WAV files are located
dir_path <- "\\clips"

# Get a list of all WAV files in the directory
wav_files <- list.files(dir_path, pattern = "\\.wav$", full.names = TRUE)

# Initialize an empty data frame to store the extracted features
features_df <- data.frame()

# Loop through each WAV file and extract the desired features
for (wav_file in wav_files) {
  
  # Load the WAV file using the tuneR package
  wave <- readWave(wav_file)
  
  # Compute the spectral properties using the seewave package
  specprop <- specprop(wave)
  
  # Compute the other desired features
  meanfreq <- mean(specprop$freq)/1000  # Convert from Hz to kHz
  sd <- sd(specprop$freq)/1000
  median <- median(specprop$freq)/1000
  Q25 <- quantile(specprop$freq, 0.25)/1000
  Q75 <- quantile(specprop$freq, 0.75)/1000
  IQR <- IQR(specprop$freq)/1000
  skew <- skewness(specprop$freq)
  kurt <- kurtosis(specprop$freq)
  sp.ent <- entropy(specprop$dB)
  sfm <- spectralFlatness(wave)
  mode <- specprop$freq[which.max(specprop$dB)]/1000
  centroid <- centroid(specprop$freq, specprop$dB)/1000
  peakf <- peakf(specprop$freq, specprop$dB)/1000
  meanfun <- mean(pitch(wave))
  minfun <- min(pitch(wave))
  maxfun <- max(pitch(wave))
  meandom <- mean(dominant(wave))
  mindom <- min(dominant(wave))
  maxdom <- max(dominant(wave))
  dfrange <- maxdom - mindom
  modindx <- modindx(wave)
  
  # Combine the features into a single row of the data frame
  row <- data.frame(
    meanfreq = meanfreq,
    sd = sd,
    median = median,
    Q25 = Q25,
    Q75 = Q75,
    IQR = IQR,
    skew = skew,
    kurt = kurt,
    sp.ent = sp.ent,
    sfm = sfm,
    mode = mode,
    centroid = centroid,
    peakf = peakf,
    meanfun = meanfun,
    minfun = minfun,
    maxfun = maxfun,
    meandom = meandom,
    mindom = mindom,
    maxdom = maxdom,
    dfrange = dfrange,
    modindx = modindx
  )
  
  # Append the row to the data frame
  features_df <- rbind(features_df, row)
}

# Write the data frame to a CSV file
write.csv(features_df, "clip_features.csv", row.names = FALSE)
