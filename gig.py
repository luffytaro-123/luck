def p1() -> str:
    return """
---------- # Aim: To understand the basic concept of optimization by finding the minimum value of a mathematical function. ----------


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 3)**2 + 2

x = np.linspace(-2, 8, 100)
y = f(x)

minimum_x = 3
minimum_y = f(minimum_x)

print("Minimum Point:", minimum_x)
print("Minimum Value:", minimum_y)

plt.plot(x, y)
plt.scatter(minimum_x, minimum_y, color="red")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Optimization of a Function")
plt.grid()
plt.show()
"""


def p2() -> str:
    return """
---------- # Aim: To understand the use of Python, NumPy, and Matplotlib for solving optimization problems. ----------

import numpy as np
import matplotlib.pyplot as plt

# Function
def f(x):
    return x**2 - 4*x + 5

# Generate values
x = np.linspace(-2, 6, 100)
y = f(x)

# Minimum point
min_index = np.argmin(y)

print("Minimum x =", x[min_index])
print("Minimum f(x) =", y[min_index])

# Plot
plt.plot(x, y)
plt.scatter(x[min_index], y[min_index], color="red")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function Optimization using Python")
plt.grid()
plt.show()
"""


def p3() -> str:
    """Text Preprocessing"""
    return """
---------- # Aim: To find the minimum point of a function using the first and second derivative method. ----------

import sympy as sp

x = sp.symbols('x')
f = x**2 - 6*x + 10

# First derivative
df = sp.diff(f, x)

# Critical point
point = sp.solve(df, x)[0]

# Second derivative
d2f = sp.diff(f, x, 2)

print("Critical Point =", point)
print("Second Derivative =", d2f)
print("Minimum Value =", f.subs(x, point))



---------- # Aim: To find and classify stationary points of single-variable and two-variable functions using classical optimization methods in Python ----------

import sympy as sp

x=sp.symbols('x')
f=x**3-6*x**2+9*x

d=sp.diff(f,x)
d2=sp.diff(f,x,2)

for p in sp.solve(d,x):
    print("Point:",p)
    print("Value:",f.subs(x,p))
    print("Type:",
          "Minimum" if d2.subs(x,p)>0
          else "Maximum" if d2.subs(x,p)<0
          else "Cannot decide")

x,y=sp.symbols('x y')
f=x**2+y**2-4*x-6*y

p=sp.solve([sp.diff(f,x),sp.diff(f,y)],[x,y])

print("2D Point:",p)
print("Minimum:",f.subs(p))
"""


def p4() -> str:
    return """
---------- # Aim: To find the minimum of a single-variable function using Interval Halving Method and Golden Section Search Method. ----------

import math

def f(x):
    return (x - 2)**2 + 1

# -----------------------------
# Interval Halving Method
# -----------------------------
def interval_halving(a, b, tol=0.01):
    while (b - a) > tol:
        mid = (a + b) / 2
        x1 = (a + mid) / 2
        x2 = (mid + b) / 2

        if f(x1) < f(mid):
            b = mid
        elif f(x2) < f(mid):
            a = mid
        else:
            a = x1
            b = x2

    return (a + b) / 2

# -----------------------------
# Golden Section Search
# -----------------------------
def golden_section(a, b, tol=0.01):
    r = (math.sqrt(5) - 1) / 2

    while (b - a) > tol:
        x1 = b - r * (b - a)
        x2 = a + r * (b - a)

        if f(x1) < f(x2):
            b = x2
        else:
            a = x1

    return (a + b) / 2

# Initial interval
a = 0
b = 5

x1 = interval_halving(a, b)
x2 = golden_section(a, b)

print("Interval Halving Method")
print("Minimum x =", round(x1, 3))
print("Minimum f(x) =", round(f(x1), 3))

print("\nGolden Section Method")
print("Minimum x =", round(x2, 3))
print("Minimum f(x) =", round(f(x2), 3))



---------- # Aim: To find the minimum of a single-variable function using the Fibonacci Search Method. ----------

f=lambda x:(x-3)**2+2

def fib(a,b,n):
    F=[0,1]
    for i in range(2,n+1):
        F.append(F[-1]+F[-2])

    for k in range(n-2):
        x1=a+F[n-k-2]/F[n-k]*(b-a)
        x2=a+F[n-k-1]/F[n-k]*(b-a)

        if f(x1)<f(x2): b=x2
        else: a=x1

    return (a+b)/2

x=fib(0,6,10)
print("Minimum x =",round(x,3))
print("Minimum f(x) =",round(f(x),3))
"""


def p5() -> str:
    return """
---------- # Aim: To find the minimum of a single-variable function using the Quadratic Interpolation Method and plot the function with its minimum point. ----------

import numpy as np
import matplotlib.pyplot as plt

f=lambda x:(x-4)**2+3

def q(x1,x2,x3):
    a,b,c=f(x1),f(x2),f(x3)
    return ((x2**2-x3**2)*a+(x3**2-x1**2)*b+
            (x1**2-x2**2)*c)/(2*((x2-x3)*a+
            (x3-x1)*b+(x1-x2)*c))

x=q(2,4,6)

print("Minimum x =",round(x,3))
print("Minimum f(x) =",round(f(x),3))

X=np.linspace(0,8,100)
plt.plot(X,f(X))
plt.scatter(x,f(x),color="red")
plt.grid()
plt.show()
"""


def p6() -> str:
    return """
---------- # Aim: To find the minimum of a single-variable function using the Quadratic Interpolation Method and plot the function with its minimum point. ----------

import numpy as np
import matplotlib.pyplot as plt

f=lambda x:x**2-6*x+10
df=lambda x:2*x-6

x=0

for i in range(5):
    x=x-df(x)/2

print("Minimum x =",round(x,3))
print("Minimum f(x) =",round(f(x),3))

X=np.linspace(-2,8,100)
plt.plot(X,f(X))
plt.scatter(x,f(x),color="red")
plt.grid()
plt.show()
"""


def p7() -> str:
    return """
---------- # Aim: To find the minimum of a function subject to an equality constraint using the Lagrange Multiplier Method and plot the result. ----------

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x,y,l=sp.symbols('x y l')

f=x**2+y**2
L=f+l*(x+y-4)

p=sp.solve([sp.diff(L,x),sp.diff(L,y),sp.diff(L,l)],[x,y,l])

print("Optimal x =",p[x])
print("Optimal y =",p[y])
print("Minimum =",f.subs(p))

X=np.linspace(0,4,100)
plt.plot(X,4-X)
plt.scatter(float(p[x]),float(p[y]),color="red")
plt.grid()
plt.show()
"""


def p8() -> str:
    return """
---------- # Aim: To find the minimum point of a function by applying the Bisection Method. ----------

import numpy as np
import matplotlib.pyplot as plt

f=lambda x:x**2-4*x+5
df=lambda x:2*x-4

a,b=0,5

for i in range(20):
    m=(a+b)/2

    if df(m)==0: break
    elif df(a)*df(m)<0: b=m
    else: a=m

print("Minimum x =",round(m,3))
print("Minimum f(x) =",round(f(m),3))

x=np.linspace(0,5,100)
plt.plot(x,f(x))
plt.scatter(m,f(m),color="red")
plt.grid()
plt.show()
"""

