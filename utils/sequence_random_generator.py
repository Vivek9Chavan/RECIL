import random

# Generate 25 random numbers
random_numbers = [random.uniform(0, 90/25) for i in range(25)]

# Scale the numbers to ensure their total is 90
total = sum(random_numbers)
random_numbers = [num * (90/total) for num in random_numbers]

print(random_numbers)
print("Total:", sum(random_numbers))