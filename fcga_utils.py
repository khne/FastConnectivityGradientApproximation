# visualization utils

import numpy as np
import matplotlib.pyplot as plt

# for brain surface plots
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# plot comparison ----------------------------------------------------------------------------------------
def visualize_comparison(grad_full, grad_fcga, grad_fcga_aligned, time_full, time_fcga, title_str=""):
    fig = plt.figure(figsize=(10,3))
    ax1 = plt.subplot(1,5,1)
    plt.bar(np.array([1,2]), np.array([time_full, time_fcga]), color=['tab:red', 'tab:blue'])
    plt.ylabel('time (sec)')
    plt.xticks(ticks=[1,2], labels=['Full FC', 'FCGA'])
    plt.title('computation time', fontsize=10)

    ax3 = plt.subplot(1,5,(2,3))
    similarity_fcga = np.array([ np.corrcoef(grad_full[:,x], grad_fcga[:,x])[0,1] for x in np.arange(25)])
    plt.bar(np.arange(25),np.abs(similarity_fcga))
    plt.ylim((0,1))
    plt.ylabel('correlation (absolute)')
    plt.xlabel('gradients')
    plt.title('spatial similarity\n(before procrustes)', fontsize=10)

    ax3 = plt.subplot(1,5,(4,5))
    similarity_fcga_aligned = np.array([ np.corrcoef(grad_full[:,x], grad_fcga_aligned[:,x])[0,1] for x in np.arange(25)])
    plt.bar(np.arange(25),np.abs(similarity_fcga_aligned))
    plt.ylim((0,1))
    plt.ylabel('correlation (absolute)')
    plt.xlabel('gradients')
    plt.title('spatial similarity\n(after procrustes to reorder)', fontsize=10)

    plt.suptitle(title_str)
    fig.tight_layout()
    fig.show()
# --------------------------------------------------------------------------------------------------------

# plot gradients -----------------------------------------------------------------------------------------
def visualize_gradients(grad_full, grad_fcga_aligned, visu_grad, surf_lh, vidx_lh):
    
    x, y, z = surf_lh['vertices'].T
    i, j, k = surf_lh['faces'].T

    gradient_full_lh = np.zeros(vidx_lh.shape)
    gradient_full_lh[vidx_lh] = grad_full[:(np.sum(vidx_lh)), visu_grad]
    norm = plt.Normalize()
    colors = plt.cm.turbo(norm(gradient_full_lh))
    surf_grad_full = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=colors, opacity=1, hoverinfo='skip')

    gradient_fcga_lh = np.zeros(vidx_lh.shape)
    gradient_fcga_lh[vidx_lh] = grad_fcga_aligned[:(np.sum(vidx_lh)), visu_grad]
    norm = plt.Normalize()
    colors = plt.cm.turbo(norm(gradient_fcga_lh))
    surf_grad_fcga_aligned = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=colors, opacity=1, hoverinfo='skip')

    #print(f"r = {np.corrcoef(gradient_full_lh,gradient_fcga_lh)[0,1]}")

    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                        subplot_titles=['Gradient - full FC', 'Gradient - FCGA'])
    fig.add_trace(surf_grad_full, row=1, col=1)
    fig.add_trace(surf_grad_fcga_aligned, row=1, col=2)

    fig.update_layout(
        height=400,
        width=700,
        scene = dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
        ),
        scene2 = dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
        )
    )

    fig.show()
# ---------------------------------------------------------------------------------------------------------
