import kagglehub

# Download latest version
path = kagglehub.dataset_download("ybifoundation/salary-prediction-simple-linear-regression")

print("Path to dataset files:", path)