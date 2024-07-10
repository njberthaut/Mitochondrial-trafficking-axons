#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def set_axes_properties(ax):
    # Define sizes
    axis_label_font_size = 10
    title_font_size = 11
    axis_tick_font_size = 10
    my_tick_length = 0.5*axis_tick_font_size
    axes_line_width = 1

    # Define labels
    x_lbl = ax.get_xlabel()
    y_lbl = ax.get_ylabel()
    ttl = ax.get_title()

    # Set default colors

    # Set axes labels and titles
    ax.xaxis.label.set_fontsize(axis_label_font_size)  # Set X-axis label font size
    ax.yaxis.label.set_fontsize(axis_label_font_size)  # Set Y-axis label font size
    ax.title.set_fontsize(title_font_size)  # Set title font size

    # Set more plot appearance
    ax.spines['bottom'].set_linewidth(axes_line_width)
    ax.spines['left'].set_linewidth(axes_line_width)
    ax.spines['right'].set_linewidth(axes_line_width)
    ax.spines['top'].set_linewidth(axes_line_width)

    ax.grid(False)  # Turn off the grid

    # Set ticks
    ax.tick_params(axis='both', which='major', length=my_tick_length, direction='out')  # Set tick size and direction

    # Set tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontsize(axis_tick_font_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(axis_tick_font_size)

    # Turn off the box around the plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# In[ ]:


def set_scatter_properties(plot_handle, clr):
    # Your custom properties
    marker_size = 40
    #marker_type = '.'
    marker_color = clr

    # Apply custom properties to the scatter plot
    plot_handle.set_sizes([marker_size])
    plot_handle.set_facecolor(marker_color)
    plot_handle.set_edgecolor(marker_color)
    #plot_handle.set_marker(marker_type) # doesn't exist with matplotlib


# In[ ]:


def set_line_properties(plot_handle, clr):
    # Your custom properties for the line plot
    line_width = 1
    line_color = clr
    line_style = '-'

    # Apply custom properties to the line plot
    plot_handle.set_linewidth(line_width)
    plot_handle.set_color(line_color)
    plot_handle.set_linestyle(line_style)


# In[ ]:


def set_figure_properties(plot_handle):
    cm = 1/2.54    
    #plt.figure(figsize=(6*cm, 6*cm))
    plot_handle.set_size_inches(8*cm, 8*cm)

