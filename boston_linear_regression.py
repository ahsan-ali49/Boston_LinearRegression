#%% Importing all the libraries for; splitting data, running linear regression and plotting graphs
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%% Loading Datasets
boston_data = pd.read_csv(r"C:\Users\Ahsan\Desktop\AIcodes\housing.csv")
#%% Seperating independent and dependent variables respectively (X=features(13), y=target(median value of home in $1000))
X = boston_data.iloc[:, :-1]
y = boston_data.iloc[:,-1]
print(boston_data.shape)

#%% Converting to list and print its dimensions
tuples = list(zip(X, y))
print(boston_data.shape)

#%% Creating seed values, and lists for MSE and Regression score
seeds = [1,2,3,4,5,6,7,8,9,10]
mse_list = []
r2_scores = []

#%% Shuffling & Spitting data 10 times, running linearRegression and predicting with R2 and MSE
for seed in seeds:
    
    # Setting seed values and shuffling
    random.seed(seed)
    random.shuffle(tuples)
    
    # Splitting data in ratio of 0.7 for training and 0.3 for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=(seed))
    print("Seed", seed, ": Train set size =", len(X_train), ", Test set size =", len(X_test))
    
    # Running LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # getting R2 score for each sample of data
    score = model.score(X_test, y_test)
    r2_scores.append(score)
    print("Seed", seed, "Regression score:", score)
    
    # getting MSE score for each sample of data
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)
    print("Seed", seed, "MSE:", mse)

#%% Plotting for MSE
plt.bar(seeds, mse_list)
plt.title("Mean Squared Error (MSE) for 10 random seeds")
plt.xlabel("Random seed")
plt.ylabel("MSE")
plt.show()

#%% Plotting for R^2 scores
plt.bar(seeds, r2_scores)
plt.title("Regression Score (R^2) for 10 random seeds")
plt.xlabel("Random seed")
plt.ylabel(("R^2"))
plt.show()

