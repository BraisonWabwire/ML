# Data from the prune log
data = [
    (21, 0, 11.59784, 22.37068),
    (20, 2.47933762, 11.79482, 22.01144),
    (19, 2.602975478, 11.92497, 21.99134),  # Minimum Validation MSE
    (18, 2.510227273, 12.16648, 22.14245),
    (17, 2.874095507, 12.28455, 22.12984),
    (16, 2.874095507, 12.42569, 22.12984),
    (15, 3.636959064, 12.653, 22.69819),
    (14, 3.277857555, 12.87976, 22.43582),
    (13, 3.277857555, 13.01188, 22.43582),
    (12, 3.166379434, 13.29182, 22.73964),
    (11, 3.359372577, 14.02543, 22.74113),
    (10, 3.819333349, 14.6908, 23.05767),
    (9, 4.60470163, 14.93437, 23.38113),
    (8, 5.988329591, 15.39484, 23.85401),
    (7, 5.868890688, 17.27776, 23.74365),
    (6, 6.68214015, 17.62498, 23.78791),
    (5, 11.29753292, 18.57957, 24.99924),
    (4, 17.4988252, 22.07933, 29.80257),
    (3, 17.11427713, 29.89755, 31.73721),
    (2, 23.45465646, 34.17612, 27.43468),
    (1, 33.43802325, 50.89513, 51.82928),
    (0, 32.671226, 83.56636, 86.50055)
]

# Convert to a list of dictionaries for easier manipulation
tree_data = [
    {"Decision Node": d, "Cost Complexity": c, "Train MSE": train, "Validation MSE": val}
    for d, c, train, val in data
]

# Find the decision node with the lowest validation MSE
best_node = min(tree_data, key=lambda x: x["Validation MSE"])

# Print result
print("Best-Pruned Tree:")
print(f"Decision Node: {best_node['Decision Node']}")
print(f"Validation MSE: {best_node['Validation MSE']:.5f}")
print(f"Training MSE: {best_node['Train MSE']:.5f}")
print(f"Cost Complexity: {best_node['Cost Complexity']:.8f}")
