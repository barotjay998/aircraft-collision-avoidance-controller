class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, cost):
        self.edges.setdefault(from_node, []).append((to_node, cost))


def assign_costs(graph, deviations):
    for (from_node, to_node), deviation in deviations.items():
        cost = calculate_cost(deviation)  # Define `calculate_cost` based on your criteria
        graph.add_edge(from_node, to_node, cost)
    
def calculate_cost(deviation):
    return deviation//100

import heapq
from math import sqrt
from scipy.stats import norm

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph.nodes}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        for neighbor, weight in graph.edges.get(current_node, []):
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances


def prioritize_collisions(collisions):
    # Sort collisions based on time step
    return sorted(collisions, key=lambda x: x.time_step)

def calculate_deviations(collision):
    return 500

def get_potential_collisions():
    return []

sigma1 = 0.3
sigma2 = 0.2
# Parameters for the relative position normal distribution
mu_r = 0  # for simplicity, assuming the expected paths cross
sigma_r = sqrt(sigma1**2 + sigma2**2)  # assuming independence

# Collision radius
R = 100  # the defined minimum separation distance

# Calculate the collision probability
collision_probability = 2 * norm.cdf(R, mu_r, sigma_r) - 1


# Main Function
def main():
    graph = Graph()
    collisions = get_potential_collisions()  # Define this function to get potential collisions
    prioritized_collisions = prioritize_collisions(collisions)

    for collision in prioritized_collisions:
        # Process each collision
        deviations = calculate_deviations(collision)  # Define this function based on your criteria
        assign_costs(graph, deviations)

        # Find the shortest path for each airplane involved in the collision
        for airplane in collision.airplanes:
            shortest_path = dijkstra(graph, airplane)
            # Update the path of the airplane based on `shortest_path`

if __name__ == "__main__":
    main()
