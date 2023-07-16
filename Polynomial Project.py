#!/usr/bin/env python
# coding: utf-8

# In[6]:


def sn(list):
    n = len(list)
    temp = []
    for i in range(n - 1):
        temp.append(list[i + 1] - list[i])
    return temp

def rsn(list, other):
    n = len(list) - 1
    temp = []
    for a in range(n + 1):
        num = list[a]
        for i in range(a):
            num -= temp[i] * comb(n + 1 - i, a +1 - i)
        num /= n + 1 - a
        temp.append(num)
    temp.append(other)
    return temp

def sn_mat(list):
    n = len(list)
    temp1 = []
    temp2 = list
    for i in range(n):
        temp1.append(temp2)
        temp2 = sn(temp2)
    return temp1

def convert_to_func(list):
    n = len(list)
    mat = sn_mat(list)
    coef = []
    for i in range(n):
        coef.append(mat[i][0])
    value = mat[n - 1]
    temp = value
    for j in range(1, n):
        i = n - 1 - j
        temp = rsn(temp, coef[i] - temp[j - 1])
    return temp

exam1 = [1, 4, 9, 16]
exam2 = [2, 1]

print(sn(exam1))
print(rsn(exam2, 0))
print(sn_mat(exam1))
print(convert_to_func(exam1))


# In[2]:


def strex_vals(a, b, N):
    list = []
    for i in range(N):
        list.append(a + (b - a) * i / (N - 1))
    return list

def strey_vals(f, x_vals):
    list = []
    for val in x_vals:
        list.append(f(val))
    return list

def sets(x, y):
    n = len(x)
    string = "f(" + str(round(x[0], 5)) + ") = " + str(round(y[0], 5))
    for i in range(1, n):
        string += ", f(" + str(round(x[i], 5)) + ") = " + str(round(y[i], 5))
    print(string)

def find_first_non_zero(list):
    temp = 0
    cond = False
    for val in list:
        if(val == 0 and cond == False):
            temp += 1
        if(val != 0):
            cond = True
    return temp

def miner(x):
    temp = abs(x) / 10
    if(abs(x) < 1):
        temp *= 20
    return x - temp

def maxer(x):
    temp = abs(x) / 10
    if(abs(x) < 1):
        temp *= 20
    return x + temp

def new_coefs(x, y):
    x_func = convert_to_func(x)
    y_func = convert_to_func(y)
    m, b = x_func[len(x) - 2], x_func[len(x) - 1]
    n = len(y_func) - 1
    new = []
    for i in range(n + 1):
        k = n - i
        temp = 0
        for j in range(i + 1):
            temp += y_func[i - j] * comb(k + j, j) * (-1) ** j * b ** j / m ** (k + j)
        new.append(temp)
    return new

def quad_sums(x, y):
    m = len(x)
    b = x[m - 2]
    C = 2 * x[m - 3]
    n = len(y) - 1
    sums = []
    for i in range(n + 1):
        k = n - i
        temp = 0
        for j in range(i + 1):
            temp += y[i - j] * comb(k + j, j) * (-1) ** j * b ** j / C ** (k + j)
        sums.append(temp)
    return sums

def quad_bounds(x, y):
    x_func = convert_to_func(x)
    y_func = convert_to_func(y)
    m = len(x_func)
    n = len(y_func)
    a, b, c = x_func[m - 3], x_func[m - 2], x_func[m - 1]
    A = b ** 2 - (4 * a * c)
    B = 4 * a
    zero_point = -A / B
    mi, ma = min(x), max(x)
    if(zero_point < mi):
        mi = zero_point
    if(zero_point > ma):
        ma = zero_point
    return mi, ma


# In[3]:


def print_polynomial(list):
    n = len(list)
    temp = []
    for val in list:
        temp.append(round(val, 5))
    a = find_first_non_zero(temp)
    temp = "f(x) = "
    for i in range(n):
        c = round(list[i], 5)
        exp = n - i - 1
        if c != 0:
            if(i == a and list[a] < 0):
                temp += " -"
            if(i > a):
                if(c > 0):
                    temp += " + "
                if(c < 0):
                    temp += " - "
            if abs(c) != 1:
                temp += str(abs(c))
            if exp > 1:
                temp += "x^" + str(exp)
            if exp == 1:
                temp += "x"
            if exp == 0 and abs(c) == 1:
                temp += "1"
    print(temp)
    
def print_quadnomial(x, y):
    x_func = convert_to_func(x)
    y_func = convert_to_func(y)
    m = len(x_func)
    n = len(y_func)
    a, b, c = x_func[m - 3], x_func[m - 2], x_func[m - 1]
    A = b ** 2 - (4 * a * c)
    B = 4 * a
    C = 2 * a
    sign = " + "
    if(B < 0):
        sign = " - "
    if(A != 0):
        r = "(" + str(A) + sign + str(abs(B)) + "x)"
    if(A == 0):
        r = "(" + str(B) + "x)"
    sums = quad_sums(x_func, y_func)
    temp = []
    for val in sums:
        temp.append(round(val, 5))
    z = find_first_non_zero(temp)
    temp = "Two functions:\nf(x) = "
    for i in range(n):
        c = round(sums[i], 5)
        exp = n - i - 1
        if c != 0:
            if(i == z and sums[z] < 0):
                temp += " -"
            if(i > z):
                if(c > 0):
                    temp += " + "
                if(c < 0):
                    temp += " - "
            if abs(c) != 1:
                temp += str(abs(c))
            if exp > 0 and exp != 2:
                temp += r + "^"
                if exp % 2 == 0:
                    temp += str(exp / 2)
                if exp % 2 == 1:
                    temp += str(exp) + "/2"
            if exp == 2:
                temp += r
            if exp == 0 and abs(c) == 1:
                temp += "1"
    temp += "\nf(x) = "
    for i in range(n):
        exp = n - i - 1
        c = round(sums[i], 5) * (-1) ** exp
        if c != 0:
            if(i == z and sums[z] > 0 and exp % 2 == 1):
                temp += " -"
            if(i > z):
                if(c > 0):
                    temp += " + "
                if(c < 0):
                    temp += " - "
            if abs(c) != 1:
                temp += str(abs(c))
            if exp > 0 and exp != 2:
                temp += r + "^"
                if exp % 2 == 0:
                    temp += str(exp / 2)
                if exp % 2 == 1:
                    temp += str(exp) + "/2"
            if exp == 2:
                temp += r
            if exp == 0 and abs(c) == 1:
                temp += "1"
    print(temp)
    
def coef_poly_value(list, x):
    n = len(list)
    temp = 0
    for i in range(n):
        exp = n - i - 1
        temp += list[i] * (x ** exp)
    return temp
    
def polynomial_value(list, x):
    coefficients = convert_to_func(list)
    n = len(list)
    temp = 0
    for i in range(n):
        exp = n - i - 1
        temp += coefficients[i] * (x ** exp)
    return temp

def quadnomial_value_p(x, y, value):
    x_func = convert_to_func(x)
    y_func = convert_to_func(y)
    m = len(x_func)
    n = len(y_func)
    a, b, c = x_func[m - 3], x_func[m - 2], x_func[m - 1]
    A = b ** 2 - (4 * a * c)
    B = 4 * a
    C = 2 * a
    r = A + (B * value)
    sums = quad_sums(x_func, y_func)
    sum = 0
    for i in range(n):
        exp = (n - 1 - i) / 2
        sum += sums[i] * r ** exp
    return sum

def quadnomial_value_n(x, y, value):
    x_func = convert_to_func(x)
    y_func = convert_to_func(y)
    m = len(x_func)
    n = len(y_func)
    a, b, c = x_func[m - 3], x_func[m - 2], x_func[m - 1]
    A = b ** 2 - (4 * a * c)
    B = 4 * a
    C = 2 * a
    r = A + (B * value)
    sums = quad_sums(x_func, y_func)
    sum = 0
    for i in range(n):
        exp = (n - 1 - i) / 2
        sum += (-1) ** (2 * exp) * sums[i] * r ** exp
    return sum


# In[4]:


from matplotlib import pyplot as plt
N_graph = 100

def polynomial_graph(list):
    a = 1
    b = len(list)
    y_func = convert_to_func(list)
    sets(strex_vals(a, b, b), list)
    print("\n")
    print_polynomial(y_func)
    x_vals = []
    y_vals = []
    for i in range(N_graph + 1):
        x_vals.append(a + (i * (b - a) / N_graph))
        y_vals.append(polynomial_value(list, x_vals[i]))
    plt.plot(x_vals, y_vals, c = 'b')
    for i in range(b):
        plt.scatter(i + 1, list[i], c = 'r')
    if(min(y_vals) < min(list) - 100 or max(y_vals) > max(list) + 100):
        plt.ylim([miner(min(list)), maxer(max(list))])
    plt.show()
    
def polynomial_graph_value(list, x):
    a = 1
    b = len(list)
    y_func = convert_to_func(list)
    sets(strex_vals(a, b, b), list)
    print("\n")
    print_polynomial(y_func)
    val = round(polynomial_value(list, x), 5)
    print("f(" + str(round(x, 5)) + ") = " + str(val))
    x_vals = []
    y_vals = []
    for i in range(N_graph + 1):
        x_vals.append(a + (i * (b - a) / N_graph))
        y_vals.append(polynomial_value(list, x_vals[i]))
    plt.plot(x_vals, y_vals, c = 'b')
    plt.scatter(x, val, c = 'g')
    for i in range(b):
        plt.scatter(i + 1, list[i], c = 'r')
    if(min(y_vals) < min(list) - 100 or max(y_vals) > max(list) + 100):
        plt.ylim([miner(min(list)), maxer(max(list))])
    plt.show()
    
def polynomial_graph_inputs(x, y):
    x_func = convert_to_func(x)
    y_func = convert_to_func(y)
    m = len(x)
    n = len(y)
    sets(x, y)
    print("\n")
    if(round(x_func[m - 3], 5) == 0):
        a, b = min(x), max(x)
        coefs = new_coefs(x, y)
        print_polynomial(coefs)
        x_vals = []
        y_vals = []
        for i in range(N_graph + 1):
            x_vals.append(a + (i * (b - a) / N_graph))
            y_vals.append(coef_poly_value(coefs, x_vals[i]))
        plt.plot(x_vals, y_vals, c = 'b')
        for i in range(len(x)):
            plt.scatter(x[i], y[i], c = 'r')
        if(min(y_vals) < min(y) - 100 or max(y_vals) > max(y) + 100):
            plt.ylim([miner(min(y)), maxer(max(y))])
        plt.show()
    if(round(x_func[m - 3], 5) != 0):
        a, b = quad_bounds(x, y)
        print_quadnomial(x, y)
        x_vals = []
        y_vals_p = []
        y_vals_n = []
        for i in range(N_graph + 1):
            x_vals.append(a + (i * (b - a) / N_graph))
            y_vals_p.append(quadnomial_value_p(x, y, x_vals[i]))
            y_vals_n.append(quadnomial_value_n(x, y, x_vals[i]))
        plt.plot(x_vals, y_vals_p, c = 'b')
        plt.plot(x_vals, y_vals_n, c = 'b')
        for i in range(len(x)):
            plt.scatter(x[i], y[i], c = 'r')
        plt.ylim([-30, 100])
        plt.show()
    
def polynomial_graph_inputs_value(x, y, value):
    x_func = convert_to_func(x)
    y_func = convert_to_func(y)
    m = len(x)
    n = len(y)
    sets(x, y)
    print("\n")
    if(round(x_func[m - 3], 5) == 0):
        a, b = min(x), max(x)
        coefs = new_coefs(x, y)
        print_polynomial(coefs)
        val = round(coef_poly_value(coefs, value), 5)
        print("f(" + str(round(value, 5)) + ") = " + str(val))
        x_vals = []
        y_vals = []
        for i in range(N_graph + 1):
            x_vals.append(a + (i * (b - a) / N_graph))
            y_vals.append(coef_poly_value(coefs, x_vals[i]))
        plt.plot(x_vals, y_vals, c = 'b')
        plt.scatter(value, val, c = 'g')
        for i in range(len(x)):
            plt.scatter(x[i], y[i], c = 'r')
        if(min(y_vals) < min(y) - 100 or max(y_vals) > max(y) + 100):
            plt.ylim([miner(min(list)), maxer(max(list))])
        plt.show()
    if(round(x_func[m - 3], 5) != 0):
        a, b = quad_bounds(x, y)
        print_quadnomial(x, y)
        val_p = round(quadnomial_value_p(x, y, value), 5)
        val_n = round(quadnomial_value_n(x, y, value), 5)
        string = "f(" + str(round(value, 5)) + ") = " + str(val_p)
        if(val_p != val_n):
            string += ", " + str(val_n)
        print(string)
        x_vals = []
        y_vals_p = []
        y_vals_n = []
        for i in range(N_graph + 1):
            x_vals.append(a + (i * (b - a) / N_graph))
            y_vals_p.append(quadnomial_value_p(x, y, x_vals[i]))
            y_vals_n.append(quadnomial_value_n(x, y, x_vals[i]))
        plt.plot(x_vals, y_vals_p, c = 'b')
        plt.plot(x_vals, y_vals_n, c = 'b')
        plt.scatter(value, val_p, c = 'g')
        plt.scatter(value, val_n, c = 'g')
        for i in range(len(x)):
            plt.scatter(x[i], y[i], c = 'r')
        plt.ylim([-30, 100])
        plt.show()


# In[11]:


pattern1 = [4, 8, 15, 16, 23, 42]

polynomial_graph(pattern1)


# In[12]:


pattern2 = [0, 2, 3, 5, 6, 8, 9]

polynomial_graph_value(pattern2, 4.5)


# In[34]:


lin1 = [3, 5, 7, 9]
lin2 = [0, 1, 4, 9]

polynomial_graph(lin1)
polynomial_graph(lin2)
polynomial_graph_inputs(lin1, lin2)


# In[33]:


lin3 = [4, -6, -16, -26]
lin4 = [6, 17, 22, 4]

polynomial_graph_inputs_value(lin3, lin4, -8.8)


# In[37]:


quad1 = [4, 9, 16, 25]
quad2 = [1, 8, 27, 64]

polynomial_graph(quad1)
polynomial_graph(quad2)
polynomial_graph_inputs(quad1, quad2)


# In[42]:


quad3 = [8, 15, 6, -19]
quad4 = [30, 22, 75, 43]

polynomial_graph_inputs_value(quad3, quad4, 12)


# In[5]:


import math

def fact(x):
    if x < 0:
        print("Error")
        return
    if x == 0:
        return 1
    return x * fact(x - 1)

def comb(n, k):
    return fact(n) / (fact(k) * fact(n - k))

def sin(x):
    return math.sin(x)

def cos(x):
    return math.cos(x)

def absolute(x):
    return abs(x)

def square(x):
    return x ** 2

def expo(x):
    return math.exp(x)

def explo(x):
    return x ** x

p = math.pi
c = 2 * math.pi

exam_x = strex_vals(0, c, 20)
exam_y_sin = strey_vals(sin, exam_x)
exam_y_cos = strey_vals(cos, exam_x)
exam_y_expo = strey_vals(expo, exam_x)

polynomial_graph_inputs_value(exam_x, exam_y_sin, p / 4)
polynomial_graph_inputs_value(exam_x, exam_y_cos, p / 4)
polynomial_graph_inputs_value(exam_x, exam_y_expo, p / 4)


# In[40]:


import random

def random_vals(a, b, N):
    list = []
    for i in range(N):
        list.append(a + (b - a) * random.random())
    return list

ran = random_vals(1, 100, 12)

polynomial_graph(ran)


# In[15]:


def harmonic(x):
    sum = 0
    for i in range(x):
        sum += 1 / (i + 1)
    return sum

def harmonic_vals(n):
    temp = []
    for i in range(n):
        temp.append(harmonic(i+1))
    return temp

form = [1, 1+1/2, 11, 50, 274, 1764, 13068]

print(harmonic_vals(5))
polynomial_graph(harmonic_vals(21))


# In[ ]:




