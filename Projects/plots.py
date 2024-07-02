import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import datetime
import pickle
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

print(datetime.datetime.now())

data = pd.read_csv('data.csv')

X1 = data[['S1']].values
y1 = data[['S9']].values

X2 = data[['S2']].values
y2 = data[['S10']].values

X3 = data[['S3']].values
y3 = data[['S11']].values

X4 = data[['S4']].values
y4 = data[['S12']].values

X5 = data[['S5']].values
y5 = data[['S13']].values

X6 = data[['S6']].values
y6 = data[['S14']].values



LR1 = LinearRegression().fit(X1, y1)
y1_pred = LR1.predict(X1)

LR2 = LinearRegression().fit(X2, y2)
y2_pred = LR2.predict(X2)

LR3 = LinearRegression().fit(X3, y3)
y3_pred = LR3.predict(X3)

LR4 = LinearRegression().fit(X4, y4)
y4_pred = LR4.predict(X4)

LR5 = LinearRegression().fit(X5, y5)
y5_pred = LR5.predict(X5)

LR6 = LinearRegression().fit(X6, y6)
y6_pred = LR6.predict(X6)



print(r2_score(y1, y1_pred))
print(r2_score(y2, y2_pred))
print(r2_score(y3, y3_pred))
print(r2_score(y4, y4_pred))
print(r2_score(y5, y5_pred))
print(r2_score(y6, y6_pred))


fig = plt.figure(figsize=(16, 8))
spec = gridspec.GridSpec(2, 4)
spec.update(wspace=0.4, hspace=0.1)

ax1 = fig.add_subplot(spec[0, 0], box_aspect=1)
ax2 = fig.add_subplot(spec[0, 1], box_aspect=1)
ax3 = fig.add_subplot(spec[0, 2], box_aspect=1)
ax4 = fig.add_subplot(spec[0, 3], box_aspect=1)
ax5 = fig.add_subplot(spec[1, 0], box_aspect=1)
ax6 = fig.add_subplot(spec[1, 1], box_aspect=1)


sc = ax1.scatter(X1, y1, color='blue', marker='.', s=24)
ax1.plot(X1, y1_pred, color='black', linewidth=3)
ax1.set_xlabel('Actual Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
ax1.set_ylabel('Predicted Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
plt.text(0.01,0.99, "R\N{SUPERSCRIPT TWO} = 0.87", transform=ax1.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.01,0.88, "RMSE = 8.92", transform=ax1.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.99,0.1, "MLP", transform=ax1.transAxes, va = "top", ha="right", fontname = 'Arial', fontsize=12, fontweight=2)
line1 = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
transform = ax1.transAxes
line1.set_transform(transform)
ax1.add_line(line1)

ax2.scatter(X2, y2, color='blue', marker='.', s=24)
ax2.plot(X2, y2_pred, color='black', linewidth=3)
ax2.set_xlabel('Actual Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
ax2.set_ylabel('Predicted Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
plt.text(0.01,0.99, "R\N{SUPERSCRIPT TWO} = 0.83", transform=ax2.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.01,0.88, "RMSE = 15.64", transform=ax2.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.99,0.1, "RF", transform=ax2.transAxes, va = "top", ha="right", fontname = 'Arial', fontsize=12, fontweight=2)
line2 = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
transform = ax2.transAxes
line2.set_transform(transform)
ax2.add_line(line2)

ax3.scatter(X3, y3, color='blue', marker='.', s=24)
ax3.plot(X3, y3_pred, color='black', linewidth=3)
ax3.set_xlabel('Actual Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
ax3.set_ylabel('Predicted Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
plt.text(0.01,0.99, "R\N{SUPERSCRIPT TWO} = 0.75", transform=ax3.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.01,0.88, "RMSE = 19.23", transform=ax3.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.99,0.1, "GB", transform=ax3.transAxes, va = "top", ha="right", fontname = 'Arial', fontsize=12, fontweight=2)
line3 = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
transform = ax3.transAxes
line3.set_transform(transform)
ax3.add_line(line3)

ax4.scatter(X4, y4, color='blue', marker='.', s=24)
ax4.plot(X4, y4_pred, color='black', linewidth=3)
ax4.set_xlabel('Actual Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
ax4.set_ylabel('Predicted Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
plt.text(0.01,0.99, "R\N{SUPERSCRIPT TWO} = 0.78", transform=ax4.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.01,0.88, "RMSE = 12.11", transform=ax4.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.99,0.1, "SVR", transform=ax4.transAxes, va = "top", ha="right", fontname = 'Arial', fontsize=12, fontweight=2)
line4 = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
transform = ax4.transAxes
line4.set_transform(transform)
ax4.add_line(line4)

ax5.scatter(X5, y5, color='blue', marker='.', s=24)
ax5.plot(X5, y5_pred, color='black', linewidth=3)
ax5.set_xlabel('Actual Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
ax5.set_ylabel('Predicted Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
plt.text(0.01,0.99, "R\N{SUPERSCRIPT TWO} = 0.69", transform=ax5.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.01,0.88, "RMSE = 35.62", transform=ax5.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.99,0.1, "DT", transform=ax5.transAxes, va = "top", ha="right", fontname = 'Arial', fontsize=12, fontweight=2)
line5 = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
transform = ax5.transAxes
line5.set_transform(transform)
ax5.add_line(line5)

ax6.scatter(X6, y6, color='blue', marker='.', s=24)
ax6.plot(X6, y6_pred, color='black', linewidth=3)
ax6.set_xlabel('Actual Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
ax6.set_ylabel('Predicted Pollutants (\N{GREEK SMALL LETTER MU}g/m\N{SUPERSCRIPT THREE})', fontsize=12, fontweight=2, fontname = 'Arial')
plt.text(0.01,0.99, "R\N{SUPERSCRIPT TWO} = 0.76", transform=ax6.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.01,0.88, "RMSE = 17.32", transform=ax6.transAxes, va = "top", ha="left", fontname = 'Arial', fontsize=12, fontweight=2)
plt.text(0.99,0.1, "kNN", transform=ax6.transAxes, va = "top", ha="right", fontname = 'Arial', fontsize=12, fontweight=2)
line6 = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
transform = ax6.transAxes
line6.set_transform(transform)
ax6.add_line(line6)



fig.savefig("Scatter.png", format='png', dpi=900)

#plt.show()

print(datetime.datetime.now())

