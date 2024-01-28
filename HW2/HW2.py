import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

# ---------------------------------------- Exercise 1: Loading Data via Python ----------------------------------------
# Reading csv file for female data
with open("female_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # normalize the number in female_stature_mm by dividing them with 1000
    # normalize that of female_bmi by dividing them with 10.
    # skip the first row
    next(reader)

    # Store modified rows in a list
    modified_rows_female = []
    for row in reader:
        row[2] = float(row[2]) / 100
        row[1] = float(row[1]) / 10
        modified_rows_female.append(row)
    
    # Use the list in subsequent iterations
    counter = 0
    for row in modified_rows_female:
        if counter == 10:
            break
        print(row[1])
        counter += 1

    counter = 0
    for row in modified_rows_female:
        if counter == 10:
            break
        print(row[2])
        counter += 1
csv_file.close()

# Reading csv file for male data
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # normalize the number in male_stature_mm by dividing them with 1000
    # normalize that of male_bmi by dividing them with 10.
    # skip the first row
    next(reader)

    # Store modified rows in a list
    modified_rows_male = []
    for row in reader:
        row[2] = float(row[2]) / 100
        row[1] = float(row[1]) / 10
        modified_rows_male.append(row)
    
    # Use the list in subsequent iterations
    counter = 0
    for row in modified_rows_male:
        if counter == 10:
            break
        print(row[1])
        counter += 1

    counter = 0
    for row in modified_rows_male:
        if counter == 10:
            break
        print(row[2])
        counter += 1  
csv_file.close()


# ---------------------------------------- Exercise 2: Build a Linear Classifier via Optimization ----------------------------------------


