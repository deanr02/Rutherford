import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
df = pd.read_csv('disc.csv')  # Replace 'data.csv' with your filename

# Extract columns
x = df.iloc[1:, 0]  # First column as x-axis
y1 = df.iloc[1:, 1]  # Second column as first y dataset
y2 = df.iloc[1:, 2]  # Third column as second y dataset
y3 = df.iloc[1:, 3]  # Fourth column as third y dataset
y_err = df.iloc[1:, 4]  # Fifth column as x errors
x_err = df.iloc[1:, 5]  # Sixth column as y errors


# Create scatter plot with error bars
plt.errorbar(x, y1, xerr=x_err, yerr=y_err, fmt='o', color='red', alpha=0.5, label='Pinhole Open, Shutters Closed')
plt.errorbar(x, y2, xerr=x_err, yerr=y_err, fmt='o', color='blue', alpha=0.5, label='Pinhole Closed, Shutters Closed')
plt.errorbar(x, y3, xerr=x_err, yerr=y_err, fmt='o', color='green', alpha=0.5, label='Pinhole Closed, Shutters Open')

plt.plot(x,y1, color = 'red', linestyle='dotted')
plt.plot(x,y2, color='blue', linestyle='dotted')
plt.plot(x,y3, color ='green', linestyle='dotted')

m1, b1 = np.polyfit(x[9:25], y1[9:25], 1)

m2, b2 = np.polyfit(x[9:25], y2[9:25], 1)
#m3, b3 = np.polyfit(x[9:20], y3[9:20], 1)

# Create the plot
plt.plot(x[9:25], m1*x[9:25] + b1, color='red', label='y = ' + str(np.round(m1,2)) + 'x + ' + str(np.round(b1,2)), linestyle='-')
plt.plot(x[9:25], m2*x[9:25] + b2, color='blue', label='y = ' + str(np.round(m2,2)) + 'x + ' + str(np.round(b2,2)), linestyle='-')



# Labels and legend
plt.xlabel('Setting (V)', fontsize=14)
plt.ylabel('Rate (Hz)', fontsize=14)
plt.title('Discriminator Setting vs Alpha Detection Rate', fontsize=16)
plt.legend(fontsize=12)

# Enable major and minor gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()

# Show plots
plt.show()