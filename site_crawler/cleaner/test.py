import csv

with open('1.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # positive
        if row['label'] == '-1':
            print(row)
        # negative
        elif row['label'] == '1':
            print(row)
        # neutral / irrelevant
        elif row['label'] == '0':
            print(row)

