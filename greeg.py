import numpy as np

# 10
# 5
# 4 5 6 3 2
c = int(input())
n = int(input())
l = list(map(int, input().split()))
k = [[k_i] for k_i in l]
k_old = [[0] for k_i in l]
print(k)
while k_old != k:
    for k_i in k:
        for c_i in k_i:
