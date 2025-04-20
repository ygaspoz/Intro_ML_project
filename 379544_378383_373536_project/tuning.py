from src.tuning.logistic_regression_tuning import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

parameter = 'max_iters'  # Choose between 'max_iters' and 'lr'

data = evaluate_hyper_parameter(parameter, 100, 5000, 100, 0.00001)

# Extract data for plotting
x = list(data.keys())
train_accuracy = [data[key][0]['train_accuracy'] for key in x]
test_accuracy = [data[key][0]['test_accuracy'] for key in x]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, train_accuracy, label='Train Accuracy', marker='o')
plt.plot(x, test_accuracy, label='Test Accuracy', marker='o')

# Add labels, title, and legend
plt.xlabel(parameter)
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracy vs ' + parameter)
plt.legend()
plt.grid(True)

# Show plot
plt.savefig('accuracy_plot_0.00001.png')