import random

list=[]
for i in range(10):
    x = random.randint(1,100)
    list.append(x)
    i +=1

print(list)
print('max:', max(list))
print('min:', min(list))

sum_score = sum(list) - max(list) -min(list)
output = sum_score/8


print('final_score=', output)