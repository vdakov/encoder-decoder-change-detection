import matplotlib.pyplot as plt 
from matplotlib.pyplot import font_manager, rcParams
import numpy as np


'''
Script meant to create a legend figure for the original paper. To reduce the visual clutter in the 
original bachelor thesis paper (see the README), I made one large figure for all fusion architectures
and a shared legend with a consistent color scheme. 
'''


font_path = 'Times New Roman.ttf'
try:
    prop = font_manager.FontProperties(fname=font_path)
    font_manager.fontManager.addfont(font_path)
    rcParams['font.family'] = prop.get_name()
except Exception as e:
    print(f"Warning: Could not load custom font. Using default font. Error: {e}")
    
n = 4

# Create a list of n distinct colors
arr = plt.cm.viridis(np.linspace(0, 1, n))


colors = {
    "FC-EF": arr[0], 
    "FC-Siam-Conc.":  arr[1], 
    "FC-Siam-Diff.": arr[2], 
    "FC-LF": arr[3],
    "FC-EF-Val.": arr[0], 
    "FC-Siam-Conc.-Val.": arr[1], 
    "FC-Siam-Diff.-Val.": arr[2], 
    "FC-LF-Val.": arr[3]
}

custom_lines = [
    plt.Line2D([0], [0], color=colors["FC-EF"], lw=2, label="_FC-EF"),
    plt.Line2D([0], [0], color=colors["FC-Siam-Conc."], lw=2, label="_FC-Siam-conc."),
    plt.Line2D([0], [0], color=colors["FC-Siam-Diff."], lw=2, label="_FC-Siam-diff."),
    plt.Line2D([0], [0], color=colors["FC-LF"], lw=2, label="_FC-LF"),
    plt.Line2D([0], [0], color='black', lw=2, label="_Ground Truth"),
    plt.Line2D([0], [0], color=colors["FC-EF-Val."], lw=2, linestyle='--', label="FC-EF-Val."),
    plt.Line2D([0], [0], color=colors["FC-Siam-Conc.-Val."], lw=2, linestyle='--', label="FC-Siam-Conc.-Val."),
    plt.Line2D([0], [0], color=colors["FC-Siam-Diff.-Val."], lw=2, linestyle='--', label="FC-Siam-Diff.-Val."),
    plt.Line2D([0], [0], color=colors["FC-LF-Val."], lw=2, linestyle='--', label="FC-LF-Val.")
    
]

fig, ax = plt.subplots()
ax.axis('off')  # Hide the axis

# Add the legend
legend = ax.legend(custom_lines, 
                   list(colors.keys())[:4] + ['Ground Truth'] + list(colors.keys())[4:] ,
                   loc='center', 
                   ncol=2)

# Adjust the figure size to fit the legend
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.set_size_inches(bbox.width * 1.25, bbox.height * 1.25)

# Save the figure
plt.savefig('legend.png', bbox_inches='tight', pad_inches=0)
