"""  Module with figure functions for making and saving graphs,
     and functions for logging results
""" 

""" Credits: the base of this code is taken from Aurelion 
Geron's Hands on Machine Learning Repo
"""


# general imports
import os 

# imports for plotting things 
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
# Change the label and tick size 
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."

def save_fig(fig_id, tight_layout=True):
    """ 
    function for saving matplotlib figures 
    fig_id is a string name for the figure
    """
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
# Animate stuff 
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,
    
def plot_animation(frames, repeat=False, interval=40):
    """
    Function for animating a list of given frames 
    frames is a list containing the frames to be animated 
    """
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)
    
# setup function for opening a log file and saving it 

# how to open and save the file in a different directory?

