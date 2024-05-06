import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"S:\Naresh IT\1st may 2024\Prakash sir work\1.POLYNOMIAL REGRESSION\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=1)  # Define the degree of polynomial

X_poly=poly_reg.fit_transform(X)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg1=PolynomialFeatures(degree=2)  # Define the degree of polynomial

X_poly1=poly_reg1.fit_transform(X)

lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly1, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg2=PolynomialFeatures(degree=3)  # Define the degree of polynomial

X_poly2=poly_reg2.fit_transform(X)

lin_reg_4=LinearRegression()
lin_reg_4.fit(X_poly2, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg3=PolynomialFeatures(degree=4)  # Define the degree of polynomial

X_poly3=poly_reg3.fit_transform(X)

lin_reg_5=LinearRegression()
lin_reg_5.fit(X_poly3, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg4=PolynomialFeatures(degree=5)  # Define the degree of polynomial

X_poly4=poly_reg4.fit_transform(X)

lin_reg_6=LinearRegression()
lin_reg_6.fit(X_poly4, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg5=PolynomialFeatures(degree=6)  # Define the degree of polynomial

X_poly5=poly_reg5.fit_transform(X)

lin_reg_7=LinearRegression()
lin_reg_7.fit(X_poly5, y)

# Plotting the results

import matplotlib.pyplot as plt

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Plot Linear Regression
axs[0, 0].scatter(X, y, marker='*', color='magenta')
axs[0, 0].plot(X, lin_reg.predict(X), color='green')
axs[0, 0].set_title('Linear Regression for 8.5 years Experience-Degree=1')
axs[0, 0].set_xlabel('Position level')
axs[0, 0].set_ylabel('Salary')

# Plot Polynomial Regression 1
axs[0, 1].scatter(X, y, marker='*', color='magenta')
axs[0, 1].plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='green')
axs[0, 1].set_title('Polynomial Regression for 8.5 years Experience-Degree=2')
axs[0, 1].set_xlabel('Position level')
axs[0, 1].set_ylabel('Salary')

# Plot Polynomial Regression 2
axs[1, 0].scatter(X, y, marker='*', color='magenta')
axs[1, 0].plot(X, lin_reg_3.predict(poly_reg1.fit_transform(X)), color='green')
axs[1, 0].set_title('Polynomial Regression for 8.5 years Experience-Degree=3')
axs[1, 0].set_xlabel('Position level')
axs[1, 0].set_ylabel('Salary')

# Plot Polynomial Regression 3
axs[1, 1].scatter(X, y, marker='*', color='magenta')
axs[1, 1].plot(X, lin_reg_4.predict(poly_reg2.fit_transform(X)), color='green')
axs[1, 1].set_title('Polynomial Regression for 8.5 years Experience-Degree=4')
axs[1, 1].set_xlabel('Position level')
axs[1, 1].set_ylabel('Salary')

# Plot Polynomial Regression 4
axs[2, 0].scatter(X, y, marker='*', color='magenta')
axs[2, 0].plot(X, lin_reg_5.predict(poly_reg3.fit_transform(X)), color='green')
axs[2, 0].set_title('Polynomial Regression for 8.5 years Experience-Degree=5')
axs[2, 0].set_xlabel('Position level')
axs[2, 0].set_ylabel('Salary')

# Plot Polynomial Regression 5
axs[2, 1].scatter(X, y, marker='*', color='magenta')
axs[2, 1].plot(X, lin_reg_6.predict(poly_reg4.fit_transform(X)), color='green')
axs[2, 1].set_title('Polynomial Regression for 8.5 years Experience-Degree=6')
axs[2, 1].set_xlabel('Position level')
axs[2, 1].set_ylabel('Salary')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Predicting a new result with Linear Regression
print("Linear Regression Prediction on 8.5 years of experience:", lin_reg.predict([[8.5]]))

# Predicting a new result with Polynomial Regression
print("Polynomial Regression Prediction on 8.5 years of experience:", lin_reg_2.predict(poly_reg.fit_transform([[8.5]])))

# Predicting a new result with Polynomial Regression
print("Polynomial Regression Prediction on 8.5 years of experience:", lin_reg_3.predict(poly_reg1.fit_transform([[8.5]])))

# Predicting a new result with Polynomial Regression
print("Polynomial Regression Prediction on 8.5 years of experience:", lin_reg_4.predict(poly_reg2.fit_transform([[8.5]])))

# Predicting a new result with Polynomial Regression
print("Polynomial Regression Prediction on 8.5 years of experience:", lin_reg_5.predict(poly_reg3.fit_transform([[8.5]])))

# Predicting a new result with Polynomial Regression
print("Polynomial Regression Prediction on 8.5 years of experience:", lin_reg_6.predict(poly_reg4.fit_transform([[8.5]])))

# Predicting a new result with Polynomial Regression
print("Polynomial Regression Prediction on 8.5 years of experience:", lin_reg_7.predict(poly_reg5.fit_transform([[8.5]])))
