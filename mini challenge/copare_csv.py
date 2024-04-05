import csv

def compare_csv(file1, file2):
    
    with open(file1, 'r') as f:
        csv1 = list(csv.reader(f))

    
    with open(file2, 'r') as f:
        csv2 = list(csv.reader(f))

    
    if csv1 == csv2:
        return True
    else:
        return False


file1 = 'predictions_resnet18_wiithout_dense.csv'
file2 = 'predictions.csv'
result = compare_csv(file1, file2)
print(result)
