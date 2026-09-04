input_file = "breast_cancer.csv"
output_file = "breast_cancer_updated.csv"

headers = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

with open(input_file, mode="r", encoding="utf-8") as infile, open(
    output_file, mode="w", encoding="utf-8", newline=""
) as outfile:
    # Skip the empty first row
    next(infile)

    # Write the new header row
    outfile.write(",".join(headers) + "\n")

    # Copy the remaining data rows
    for line in infile:
        outfile.write(line)