import itertools
import openpyxl
import os

# Load the workbook
file_path = "data/Helpful Information in Den Haag.xlsx"
workbook = openpyxl.load_workbook(file_path, data_only=True)
sheet = workbook["Offers"]

# Read all rows into a list
rows = [((row[2], row[1]), row) for row in sheet.iter_rows(values_only=True)]

# Extract headers and data rows
headers, data_rows = rows[0], rows[1:]
data_rows = [((int(i[0][1]), int(i[0][0])), i[1]) for i in data_rows if i[0][0] is not None]

# Sort rows based on keys
data_rows.sort(key=lambda x: x[0])

# Group by the key
grouped_rows = itertools.groupby(data_rows, key=lambda x: x[0])

# Output directory
output_dir = "fastapi/context"
os.makedirs(output_dir, exist_ok=True)

# Process groups and write to files
for _, group in grouped_rows:
    first_entry = True
    file_name = None

    for entry in group:
        if first_entry:
            first_entry = False
            file_name = str(entry[1][0]).strip().replace("/", "-")
            file_path = os.path.join(output_dir, f"{file_name}.txt")

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"{entry[1][0]}\n\n")
            for idx in [6,7,9,10,11,12,13,14,15]:
                if entry[1][idx] is not None:
                    header = headers[1][idx].split("\n")[0]
                    file.write(f"{header}: {entry[1][idx]}\n")

# Close the workbook
workbook.close()