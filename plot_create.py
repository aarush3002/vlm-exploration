import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def create_exploration_plot():
    """
    Generates a scatter plot to compare the performance of three
    robotic exploration methods across six environments.
    """
    # --- Data Setup ---
    # Data is stored in a dictionary where each key is a method name.
    # Each method has a specific marker and a list of data points.
    # Each data point is a tuple: (distance_traveled, exploration_percentage, environment_id)
    plot_data = {
        'Gemini-Based': {
            'marker': 'o',
            'points': [
                (41.93, 97.47, 1), (418.26, 99.61, 2), (57.22, 99.84, 3),
                (115.33, 95.15, 4), (698.29, 98.69, 5), (143.90, 99.96, 6)
            ]
        },
        'Greedy (Explore Lite)': {
            'marker': '+',
            'points': [
                (63.10, 89.85, 1), (326.23, 84.61, 2), (53.08, 97.55, 3),
                (57.96, 83.38, 4), (535.45, 93.02, 5), (160.16, 76.76, 6)
            ]
        },
        'OpenCV + NBV': {
            'marker': '*',
            'points': [
                (53.63, 99.20, 1), (294.27, 96.67, 2), (34.82, 79.20, 3),
                (33.08, 53.27, 4), (541.13, 72.24, 5), (274.95, 98.56, 6)
            ]
        }
    }

    # Define a color for each environment
    env_colors = {
        1: 'blue', 2: 'red', 3: 'orange',
        4: 'green', 5: 'magenta', 6: 'cyan'
    }

    # --- Plotting ---
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 7))

    # Iterate through each method and its data to plot the points
    for method_name, method_info in plot_data.items():
        marker_shape = method_info['marker']
        for x, y, env_id in method_info['points']:
            ax.scatter(x, y,
                       color=env_colors[env_id],
                       marker=marker_shape,
                       s=100,  # size of the marker
                       alpha=0.8,
                       edgecolors='black',
                       linewidth=0.5)

    # --- Legends ---
    # Create a custom legend for the methods (markers)
    method_legend_handles = [
        mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                      markersize=10, label='Gemini-Based'),
        mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                      markersize=10, label='Greedy (Explore Lite)'),
        mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                      markersize=10, label='OpenCV + NBV')
    ]
    # Place the methods legend inside the plot
    method_legend = ax.legend(handles=method_legend_handles, loc='lower right', title='Methods')
    ax.add_artist(method_legend) # Add the first legend manually

    # Create a custom legend for the environments (colors)
    env_legend_handles = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                      markersize=10, label=f'Env. {env_id}')
        for env_id, color in env_colors.items()
    ]
    # Place the environment legend outside the plot area
    ax.legend(handles=env_legend_handles, title='Environments', loc='center left', bbox_to_anchor=(1.02, 0.5))

    # --- Final Touches ---
    # Set titles and labels for clarity
    ax.set_title('Comparative Performance of Exploration Methods', fontsize=16)
    ax.set_xlabel('Total Distance Traveled (m)', fontsize=12)
    ax.set_ylabel('Percentage of Environment Explored (%)', fontsize=12)

    # Set axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=40)

    # Add a grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust layout to make room for the legend outside the plot
    fig.subplots_adjust(right=0.75)
    plt.show()

# --- Run the function to create the plot ---
if __name__ == '__main__':
    create_exploration_plot()
