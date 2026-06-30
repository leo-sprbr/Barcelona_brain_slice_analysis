import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Definitions ---
datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3']
y_coords = np.arange(len(datasets))  # Vertical positions for datasets

# # Keeping only the Frequency data
# freq = [0.2, 0.643, 0.134]
# Provided experimental metrics & errors
freq = [0.789, 0.959, 0.900]

# Custom color mapping per dataset bar
colors = ['#566774', '#c13639', '#3681ba']

# --- 2. Create Figure and Layout ---
fig, ax = plt.subplots(figsize=(8, 4.5))

# Move the X-axis spine and label positions to the top
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# --- 3. Plot Horizontal Bars with Custom Colors ---
bars = ax.barh(y_coords, freq, height=0.5, color=colors, 
               edgecolor='black', linewidth=0.7, alpha=0.85)

# --- 4. Configure Labeling, Ticks, and Aesthetics ---
# Shared Vertical Dataset Axis
ax.set_ylabel('Experimental Datasets', fontsize=12, fontweight='bold', labelpad=12)
ax.set_yticks(y_coords)
ax.set_yticklabels(datasets, fontsize=11)
ax.invert_yaxis()  # Keeps Dataset 1 cleanly at the top position

# Top X-Axis Configuration
# Using a neutral slate color for the main label to balance the multi-colored bars
ax.set_xlabel('SO Frequency (Hz)', fontsize=12, fontweight='bold', color='#334155', labelpad=12)
ax.tick_params(axis='x', labelcolor='black')

# Add values at the end of each bar for immediate reading
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f} Hz', 
            va='center', ha='left', fontsize=10, fontweight='bold', color='#334155')

# --- 5. Clean Layout and Formatting Spines ---
ax.set_title('Slow Oscillation Frequency Across Datasets', fontsize=13, fontweight='bold', y=1.2)
ax.grid(True, axis='x', linestyle='--', alpha=0.3)  # Vertical grid lines to track bar length

# Hide structural side/bottom frames for a modern, minimal look
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Keep the top spine visible and set it to a neutral dark color
ax.spines['top'].set_visible(True)
ax.spines['top'].set_color('#475569')
ax.spines['top'].set_linewidth(1.5)

# Save with a transparent background
plt.savefig('so_frequency_custom_colors.png', transparent=True, dpi=300, bbox_inches='tight')

plt.show()