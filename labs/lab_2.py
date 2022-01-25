import numpy as np

# 9
# ---------1 level --------------------
a = np.random.randint(-10, 10, (5, 5))
print(a)
sum_ = 0

for i in range(len(a)):
    for j in range(len(a[0])):
        if a[i][j] < 0:
            sum_ += a[i][j]
        else:
            sum_ += a[i][j]
print(sum_)


# 1
b = np.random.randint(0, 11, (11, 3))
c = np.random.randint(0, 3, 3)

print(b)
print("----")
print(c)
print("----")
print(np.multiply(b, c))

# 3

d = np.ones((6, 6))
print(d)

f = False

sum_1 = 0

for i in range(len(d)):
    for j in range(len(d[0])):
        if d[i][j] == 1:
            sum_1 += 1
if sum_1 == len(d) * len(d[0]):
    f = True

print(f)


# ---------2 level --------------------
# 27

g = 5
arr = [0] * g
for i in range(g):
    s = [0] * (1 + i)
    for j in range(len(s)):
        if j == 0 or j == len(s) - 1:
            s[j] = 1
    arr[i] = s

for p in range(g):
    print("  " * (g - p), end="")
    for i in range(len(arr[p])):
        print(arr[p][i], end="   ")
    print("", end="\n")


# 1

v = np.random.randint(0, 10, 10)
print(v)
print(np.amax(v))



# 11

w = np.random.randint(0, 10, 10)
print(w)
N = 10
for i in range(N - 1):
    for j in range(N - i - 1):
        if w[j] % 2:
            w[j], w[j + 1] = w[j + 1], w[j]
print(w)





