from collections import defaultdict
import heapq
from matplotlib import pyplot as plt
import math
import io
from PIL import Image as PILImage

def euclidean_dist(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Vertex({self.x}, {self.y})"
    
class Graph:
    def __init__(self):
        self.adj_list = defaultdict(list)
        self.vertex_labels = {}
        self.vertex_to_counter = {}
        self.counter = 0
    
    def create_edge(self, vertex1, vertex2):
        distance = euclidean_dist(vertex1.x, vertex1.y, vertex2.x, vertex2.y)
        self.adj_list[vertex1].append((vertex2, distance))
        self.adj_list[vertex2].append((vertex1, distance))

        if self.counter == 0:
            self.vertex_labels[self.counter] = vertex1
            self.vertex_to_counter[vertex1] = self.counter
        
        self.counter += 1
        self.vertex_labels[self.counter] = vertex2
        self.vertex_to_counter[vertex2] = self.counter

    
    def dijkstra(self, start, end):
        # Priority queue for tracking minimum distances
        pq = []
        # Dictionary to store the shortest distance to each vertex
        distances = {vertex: float('inf') for vertex in self.adj_list}
        # Dictionary to store the previous vertex in the shortest path
        previous = {vertex: None for vertex in self.adj_list}

        # Start vertex distance is 0
        distances[start] = 0
        heapq.heappush(pq, (0, start))

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)

            # If we've reached the end, construct the path
            if current_vertex == end:
                path = []
                while current_vertex is not None:
                    path.append(current_vertex)
                    current_vertex = previous[current_vertex]
                return path[::-1]

            # Skip processing if we find a longer path
            if current_distance > distances[current_vertex]:
                continue

            # Explore neighbors
            for neighbor, weight in self.adj_list[current_vertex]:
                distance = current_distance + weight

                # If a shorter path to the neighbor is found
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (distance, neighbor))

        return []  # Return an empty path if no path is found
    
    def total_edge_distance(self):
        """
        Calculates the total sum of all unique edge distances in the graph.
        """
        total_distance = 0.0
        counted_edges = set()

        for vertex, neighbors in self.adj_list.items():
            for neighbor, distance in neighbors:
                # Create a unique, order-independent key for each edge
                # A frozenset is used because it's hashable and order doesn't matter
                edge = frozenset([vertex, neighbor])
                
                # If we haven't already counted this edge, add its distance
                if edge not in counted_edges:
                    total_distance += distance
                    counted_edges.add(edge)
        
        return total_distance

    def visualize(
        self, 
        robot_position=None, 
        robot_orientation=None,
        arrow_length=0.5
    ):
        """
        Creates a matplotlib figure showing:
          - Each vertex (with (x, y) label)
          - Each undirected edge (with distance label)
          - The robot's current position (if provided)
          - The robot's orientation as an arrow (if robot_position and orientation are provided)
        
        Returns:
            PIL Image object containing the plot.
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        # Collect all edges in a set so we don't double-draw edges
        # (Since the graph is undirected, each edge is stored twice)
        drawn_edges = set()

        # Plot vertices
        for vertex in self.adj_list:
            ax.scatter(vertex.x, vertex.y, color='blue')
            # Optionally, you can annotate vertices here if desired:
            # ax.annotate(f"({vertex.x:.1f}, {vertex.y:.1f})", (vertex.x, vertex.y),
            #             textcoords="offset points", xytext=(5, 5), fontsize=8, color='blue')

        # Plot edges
        for vertex in self.adj_list:
            for neighbor, distance in self.adj_list[vertex]:
                edge_key = tuple(sorted([(vertex.x, vertex.y), (neighbor.x, neighbor.y)]))
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)

                x_vals = [vertex.x, neighbor.x]
                y_vals = [vertex.y, neighbor.y]
                ax.plot(x_vals, y_vals, color='black', linestyle='-')

                # Optionally, add edge distance labels here

        # If we have a robot position, plot it
        if robot_position is not None:
            rx, ry = robot_position
            ax.scatter(rx, ry, color='red', s=100, marker='o', label='Robot')

            # Draw an arrow for orientation if provided
            if robot_orientation is not None:
                dx = arrow_length * math.cos(robot_orientation)
                dy = arrow_length * math.sin(robot_orientation)
                ax.arrow(rx, ry, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')

        ax.set_aspect('equal', 'box')
        ax.grid(True)
        ax.legend()

        # Adjust plot limits with padding
        all_x = [v.x for v in self.adj_list]
        all_y = [v.y for v in self.adj_list]
        if all_x and all_y:
            pad = 1
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

        # Convert the plot to a PIL image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        pil_image = PILImage.open(buf)
        plt.close(fig)
        return pil_image

def scale_intrinsics(original_intrinsics, new_w=32, new_h=24, old_w=320, old_h=240):
    """
    Scale the camera intrinsics if you're resizing from (old_w, old_h)
    to (new_w, new_h).
    """
    sx = new_w / float(old_w)
    sy = new_h / float(old_h)

    scaled = {}
    scaled["fx"] = original_intrinsics["focal_length"]["fx"] * sx
    scaled["fy"] = original_intrinsics["focal_length"]["fy"] * sy
    scaled["cx"] = original_intrinsics["principal_point"]["cx"] * sx
    scaled["cy"] = original_intrinsics["principal_point"]["cy"] * sy
    # Distortion coefficients are often left unchanged for basic use-cases,
    # but in principle you may need to re-estimate them if doing rigorous
    # camera calibration after resizing.
    return scaled