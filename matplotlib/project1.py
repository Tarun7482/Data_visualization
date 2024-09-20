# Import dependencies

import numpy as np
import pandas as pd
# Import Matplotlib

import matplotlib.pyplot as plt 

x1 = np.linspace(0, 10, 100)


# create a plot figure
fig = plt.figure()

plt.plot(x1, np.sin(x1), '-')
plt.plot(x1, np.cos(x1), '--');
plt.show()

# create a plot figure
plt.figure()


# create the first of two panels and set current axis
plt.subplot(2, 1, 1)   # (rows, columns, panel number)
plt.plot(x1, np.sin(x1))


# create the second of two panels and set current axis
plt.subplot(2, 1, 2)   # (rows, columns, panel number)
plt.plot(x1, np.cos(x1));
plt.show()

# create a plot figure
plt.figure()


# create the first of two panels and set current axis
plt.subplot(2, 1, 1)   # (rows, columns, panel number)
plt.plot(x1, np.sin(x1))


# create the second of two panels and set current axis
plt.subplot(2, 1, 2)   # (rows, columns, panel number)
plt.plot(x1, np.cos(x1));


plt.plot([1, 2, 3, 4])
plt.ylabel('Numbers')
plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()

x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)
plt.show()

# Call plot() method on the appropriate object
ax[0].plot(x1, np.sin(x1), 'b-')
ax[1].plot(x1, np.cos(x1), 'b-');
plt.show()

fig = plt.figure()

x2 = np.linspace(0, 5, 10)
y2 = x2 ** 2

axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x2, y2, 'r')

axes.set_xlabel('x2')
axes.set_ylabel('y2')
axes.set_title('title');
plt.show()

fig = plt.figure()

ax = plt.axes()
plt.show()
# Creating empty matplotlib figure with four subplots


fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
plt.show()

x3 = np.arange(0.0, 6.0, 0.01) 

plt.plot(x3, [xi**2 for xi in x3], 'b-') 

plt.show()

x4 = range(1, 5)

plt.plot(x4, [xi*1.5 for xi in x4])

plt.plot(x4, [xi*3 for xi in x4])

plt.plot(x4, [xi/3.0 for xi in x4])

plt.show()
#Line plot
# Create figure and axes first
fig = plt.figure()

ax = plt.axes()

# Declare a variable x5
x5 = np.linspace(0, 10, 1000)


# Plot the sinusoid function
ax.plot(x5, np.sin(x5), 'b-'); 
plt.show()

# Plot the sinusoid function 

plt.plot(x5, np.sin(x5));
plt.show()

fig = plt.figure()

ax = plt.axes()

x6 = np.linspace(0, 10, 1000)

ax.plot(x6, np.sin(x6), 'b-')

ax.plot(x6, np.cos(x6), 'r-'); 
plt.show()

#scatter plot
x7 = np.linspace(0, 10, 30)

y7 = np.sin(x7)

plt.plot(x7, y7, 'o', color = 'black');
plt.show()

plt.plot(x7, y7, '-ok');
plt.show()
#Scatter Plot with plt.scatter()
plt.scatter(x7, y7, marker='o')
plt.show()

#Histogram
data1 = np.random.randn(1000)

plt.hist(data1); 
plt.show()
plt.hist(data1, bins=30, density=True, alpha=0.5, histtype='stepfilled', color='steelblue')
plt.show()
#Compare Histograms of several distributions
x1 = np.random.normal(0, 4, 1000)
x2 = np.random.normal(-2, 2, 1000)
x3 = np.random.normal(1, 5, 1000)

kwargs = dict(histtype='stepfilled', alpha = 0.5, density = True, bins = 40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)

plt.show();
#Two-Dimensional Histograms
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x8, y8 = np.random.multivariate_normal(mean, cov, 10000).T

plt.hist2d(x8, y8, bins = 30, cmap = 'Blues')

cb = plt.colorbar()

cb.set_label('counts in bin')
plt.show()
#Multiple Bar Chart
data3 = [[15., 25., 40., 30.],
         [11., 23., 51., 17.],
         [16., 22., 52., 19.]]

z1 = np.arange(4)

plt.bar(z1 + 0.00, data3[0], color = 'r', width = 0.25)
plt.bar(z1 + 0.25, data3[1], color = 'g', width = 0.25)
plt.bar(z1 + 0.50, data3[2], color = 'b', width = 0.25)

plt.show()

# Stacked Bar Chart
A = [15., 30., 45., 22.]

B = [15., 25., 50., 20.]

z2 = range(4)

plt.bar(z2, A, color = 'b')
plt.bar(z2, B, color = 'r', bottom = A)

plt.show()
#Back-to-Back Bar Charts
U1 = np.array([15., 35., 45., 32.])
U2 = np.array([12., 30., 50., 25.])

z1 = np.arange(4)

plt.barh(z1, U1, color = 'r')
plt.barh(z1, -U2, color = 'b')

plt.show()
#Pie Chart
plt.figure(figsize=(7,7))

x10 = [35, 25, 20, 20]

labels = ['Computer', 'Electronics', 'Mechanical', 'Chemical']

plt.pie(x10, labels=labels);

plt.show()
#Exploded Pie Chart
plt.figure(figsize=(7,7))

x11 = [30, 25, 20, 15, 10]

labels = ['Computer', 'Electronics', 'Mechanical', 'Chemical', 'Agriculture']

explode = [0.2, 0.1, 0.1, 0.05, 0]

plt.pie(x11, labels=labels, explode=explode, autopct='%1.1f%%');

plt.show()
#Contour Plot
# Create a matrix
matrix1 = np.random.rand(10, 20)

cp = plt.contour(matrix1)

plt.show()

#clabel
x13 = np.arange(-2, 2, 0.01)
y13 = np.arange(-2, 2, 0.01)

X, Y = np.meshgrid(x13, y13)

ellipses = X*X/9 + Y*Y/4 - 1

cs = plt.contour(ellipses)

plt.clabel(cs)

plt.show()

#Image Plot
x13 = np.arange(-2, 2, 0.01)
y13 = np.arange(-2, 2, 0.01)

X, Y = np.meshgrid(x13, y13)

ellipses = X*X/9 + Y*Y/4 - 1

plt.imshow(ellipses);

plt.colorbar();

plt.show()
# Polar Chart
theta = np.arange(0., 2., 1./180.)*np.pi

plt.polar(3*theta, theta/5);

plt.polar(theta, np.cos(4*theta));

plt.polar(theta, [1.4]*len(theta));

plt.show()