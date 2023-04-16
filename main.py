import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

# Load the historical financial data

df = pd.read_csv('data/financial_data.csv')

# Create a graph structure that represents the company's revenue streams

nodes = list(df['Revenue Source'].unique())

edges = [(u, v) for u, v in zip(df['Revenue Source'], df['Expense']) if u != v]

# Create a graph neural network model

class GNN(nn.Module):

    def __init__(self, num_nodes, num_features, hidden_size, num_classes):

        super(GNN, self).__init__()

        self.conv1 = nn.Conv(num_features, hidden_size, kernel_size=1)

        self.conv2 = nn.Conv(hidden_size, hidden_size, kernel_size=1)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.dropout(x, p=0.5)

        x = self.fc(x)

        return x

# Create a data loader for the graph neural network

dataset = data.Dataset(nodes, edges)

dataloader = data.DataLoader(dataset, batch_size=32)

# Train the graph neural network model

model = GNN(len(nodes), df.shape[1], 128, 2)

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

for epoch in range(10):

    for batch in dataloader:

        nodes, edges = batch

        x = torch.Tensor(nodes)
        y = torch.Tensor(edges)

        y_hat = model(x, y)

        loss = criterion(y_hat, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

# Evaluate the graph neural network model

y_pred = model(torch.Tensor(nodes), torch.Tensor(edges))

y_true = torch.Tensor(edges)

accuracy = (y_pred == y_true).sum().item() / len(y_true)

print('Accuracy:', accuracy)

# Save the graph neural network model

torch.save(model.state_dict(), 'model.pth')

# Load the saved graph neural network model

model = GNN(len(nodes), df.shape[1], 128, 2)

model.load_state_dict(torch.load('model.pth'))

# Predict future financial projections

x = torch.Tensor(nodes)

y_hat = model(x)

# Print the predicted future financial projections

print('Predicted future financial projections:')

for node, value in zip(nodes, y_hat):

    print('{}: {}'.format(node, value))
    # Calculate the mean absolute error between the predicted and actual values

mae = torch.mean(torch.abs(y_hat - y_true))

print('Mean absolute error:', mae)

# Calculate the root mean squared error between the predicted and actual values

rmse = torch.sqrt(torch.mean((y_hat - y_true)**2))

print('Root mean squared error:', rmse)

# Plot the predicted and actual values

plt.plot(y_hat, label='Predicted')

plt.plot(y_true, label='Actual')

plt.legend()

plt.show()

# Analyze the model's predictions

for node, value in zip(nodes, y_hat):

    if value > y_true[node]:

        print('The model predicts that {} will increase in the future.'.format(node))

    elif value < y_true[node]:

        print('The model predicts that {} will decrease in the future.'.format(node))

    else:

        print('The model predicts that {} will remain the same in the future.'.format(node))
        # Calculate the correlation between the predicted and actual values

correlation = torch.corrcoef(y_hat, y_true)[0, 1]

print('Correlation:', correlation)

# Calculate the variance of the predicted values

variance = torch.var(y_hat)

print('Variance:', variance)

# Calculate the standard deviation of the predicted values

standard_deviation = torch.std(y_hat)

print('Standard deviation:', standard_deviation)

# Plot the distribution of the predicted values

plt.hist(y_hat)

plt.show()

# Analyze the model's sensitivity to changes in the input data

for node, value in zip(nodes, y_hat):

    change = torch.abs(y_hat[node] - y_true[node])

    sensitivity = change / y_true[node]

    print('The model is {} sensitive to changes in {}.'.format(sensitivity, node))

# Analyze the model's robustness to noise

noise = torch.randn(len(y_hat))

y_hat_noisy = y_hat + noise

# Calculate the mean absolute error between the predicted and actual values with noise

mae_noisy = torch.mean(torch.abs(y_hat_noisy - y_true))

print('Mean absolute error with noise:', mae_noisy)

# Calculate the root mean squared error between the predicted and actual values with noise

rmse_noisy = torch.sqrt(torch.mean((y_hat_noisy - y_true)**2))

print('Root mean squared error with noise:', rmse_noisy)

# Plot the predicted and actual values with noise

plt.plot(y_hat_noisy, label='Predicted with noise')

plt.plot(y_true, label='Actual')

plt.legend()

plt.show()
# Analyze the model's performance on different data sets

for data_set in ['train', 'validation', 'test']:

    y_hat = model(torch.Tensor(nodes))

    y_true = torch.Tensor(edges)

    mae = torch.mean(torch.abs(y_hat - y_true))

    rmse = torch.sqrt(torch.mean((y_hat - y_true)**2))

    print('Mean absolute error on {} data set: {}'.format(data_set, mae))

    print('Root mean squared error on {} data set: {}'.format(data_set, rmse))

# Analyze the model's performance over time

for epoch in range(10):

    y_hat = model(torch.Tensor(nodes))

    y_true = torch.Tensor(edges)

    mae = torch.mean(torch.abs(y_hat - y_true))

    rmse = torch.sqrt(torch.mean((y_hat - y_true)**2))

    print('Mean absolute error at epoch {}: {}'.format(epoch, mae))

    print('Root mean squared error at epoch {}: {}'.format(epoch, rmse))

# Analyze the model's performance on different input features

for feature in ['Revenue Source', 'Expense', 'External Market Factors']:

    y_hat = model(torch.Tensor(nodes), feature)

    y_true = torch.Tensor(edges)

    mae = torch.mean(torch.abs(y_hat - y_true))

    rmse = torch.sqrt(torch.mean((y_hat - y_true)**2))

    print('Mean absolute error on {} feature: {}'.format(feature, mae))

    print('Root mean squared error on {} feature: {}'.format(feature, rmse))
# Plot the mean absolute error and root mean squared error for each input feature

plt.plot(mae, label='Mean absolute error')

plt.plot(rmse, label='Root mean squared error')

plt.xlabel('Input feature')

plt.ylabel('Error')

plt.legend()

plt.show()

# Analyze the model's performance on different combinations of input features

for feature1 in ['Revenue Source', 'Expense']:

    for feature2 in ['Revenue Source', 'Expense']:

        if feature1 != feature2:

            y_hat = model(torch.Tensor(nodes), feature1, feature2)

            y_true = torch.Tensor(edges)

            mae = torch.mean(torch.abs(y_hat - y_true))

            rmse = torch.sqrt(torch.mean((y_hat - y_true)**2))

            print('Mean absolute error on {} and {} features: {}'.format(feature1, feature2, mae))

            print('Root mean squared error on {} and {} features: {}'.format(feature1, feature2, rmse))

# Analyze the model's performance on different combinations of input features and noise levels

for feature1 in ['Revenue Source', 'Expense']:

    for feature2 in ['Revenue Source', 'Expense']:

        if feature1 != feature2:

            for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:

                noise = torch.randn(len(y_hat)) * noise_level

                y_hat_noisy = y_hat + noise

                mae_noisy = torch.mean(torch.abs(y_hat_noisy - y_true))

                rmse_noisy = torch.sqrt(torch.mean((y_hat_noisy - y_true)**2))

                print('Mean absolute error on {} and {} features with noise level {}: {}'.format(feature1, feature2, noise_level, mae_noisy))

                print('Root mean squared error on {} and {} features with noise level {}: {}'.format(feature1, feature2, noise_level, rmse_noisy))
# Analyze the model's performance on different combinations of input features and noise levels

for feature1 in ['Revenue Source', 'Expense']:

    for feature2 in ['Revenue Source', 'Expense']:

        if feature1 != feature2:

            for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:

                noise = torch.randn(len(y_hat)) * noise_level

                y_hat_noisy = y_hat + noise

                mae_noisy = torch.mean(torch.abs(y_hat_noisy - y_true))

                rmse_noisy = torch.sqrt(torch.mean((y_hat_noisy - y_true)**2))

                print('Mean absolute error on {} and {} features with noise level {}: {}'.format(feature1, feature2, noise_level, mae_noisy))

                print('Root mean squared error on {} and {} features with noise level {}: {}'.format(feature1, feature2, noise_level, rmse_noisy))

# Create a function to plot the mean absolute error and root mean squared error for each input feature and noise level

def plot_mae_rmse(feature1, feature2, noise_levels):

    mae = []

    rmse = []

    for noise_level in noise_levels:

        noise = torch.randn(len(y_hat)) * noise_level

        y_hat_noisy = y_hat + noise

        mae.append(torch.mean(torch.abs(y_hat_noisy - y_true)))

        rmse.append(torch.sqrt(torch.mean((y_hat_noisy - y_true)**2)))

    plt.plot(noise_levels, mae, label='Mean absolute error')

    plt.plot(noise_levels, rmse, label='Root mean squared error')

    plt.xlabel('Noise level')

    plt.ylabel('Error')

    plt.legend()

    plt.show()

# Plot the mean absolute error and root mean squared error for each input feature and noise level

for feature1 in ['Revenue Source', 'Expense']:

    for feature2 in ['Revenue Source', 'Expense']:

        if feature1 != feature2:

            plot_mae_rmse(feature1, feature2, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

# End the program

print('End of program.')



