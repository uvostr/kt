arr = input().split()
count = 0
unique_arr = []

for x in arr:
    if x not in unique_arr:
        unique_arr.append(x)
        count += 1
        
print(len(unique_arr))