import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import math


class Visualisation:
    def __init__(self):
        pass

    @staticmethod
    def fitness_trend(trend):
        df_ft = pd.DataFrame(trend)
        g = sns.relplot(kind="line", data=df_ft)
        plt.show()

    @staticmethod
    def fitness_trend_all_optimizers(optimizers):
        df_ft = pd.DataFrame()
        for opt in optimizers:
            if optimizers[opt]['enabled']:
                df_ft[opt] = optimizers[opt]['ft']

        if not df_ft.empty:
            g = sns.relplot(kind="line", data=df_ft)
            plt.show()

    def gantt_schedule(self, fitness, machines, jobs):
        x_width = fitness
        if jobs['quantity'] <= 20:
            x_width += 280  # Build in margin for legend

        y_pos_max = 0
        y_ticks = []
        y_labels = []

        job_bar_height = 20
        job_legend = {}

        # Generate colour map of distinguishable colours
        distinct_colours = 5
        shades = int(jobs['quantity'] / distinct_colours)
        job_cmap = self.generate_colormap(shades * distinct_colours)

        fig, ax = plt.subplots()

        for mi, m in enumerate(machines['assigned_jobs']):
            y_pos_max = 0
            y_pos_min = 30000
            for ji, j in enumerate(m):
                y_machine_job_pos = (((mi + 1) * (jobs['quantity'] * job_bar_height)) + 100 * mi) - (ji * job_bar_height)
                if y_machine_job_pos > y_pos_max:
                    y_pos_max = y_machine_job_pos
                elif y_machine_job_pos < y_pos_min:
                    y_pos_min = y_machine_job_pos
                x_job_start = j[1]
                x_job_length = j[2] - j[1]
                ax.broken_barh([(x_job_start, x_job_length)], (y_machine_job_pos, job_bar_height),
                               facecolors=job_cmap.colors[ji], label='Job ' + str(j[0]) if ji not in job_legend else '')
                if ji not in job_legend:
                    job_legend[ji] = ji

            y_ticks.append(int(y_pos_max + y_pos_min) / 2)
            y_labels.append('Machine ' + str(mi))

        ax.set_xlabel('Seconds Since Scheduling Start', fontsize=7)

        # Set chart size limits
        ax.set_ylim(0, y_pos_max + 40)
        ax.set_xlim(-25, x_width)  # Build left padded margin to see first jobs when v. short

        # Configure ticks
        ax.grid(which='minor', alpha=0.1)
        ax.grid(which='major', alpha=0.5)
        major_ticks = np.arange(0, fitness, 200)
        minor_ticks = np.arange(0, fitness, 50)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis='x', which='major', labelsize=6)
        ax.tick_params(axis='y', which='major', labelsize=7)

        # Set legend
        font = font_manager.FontProperties(size='x-small')
        ax.legend(loc=1, prop=font, numpoints=1)

        # Disable legend when much longer than charted machine loading
        if jobs['quantity'] > 20:
            ax.legend().set_visible(False)

        plt.show()

    # Colormap generation courtesy of
    # https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours
    @staticmethod
    def generate_colormap(number_of_distinct_colors: int = 80):
        if number_of_distinct_colors == 0:
            number_of_distinct_colors = 80

        number_of_shades = 7
        number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades)
                                                                * number_of_shades)

        # Create an array with uniformly drawn floats taken from <0, 1) partition
        linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

        # We are going to reorganise monotonically growing numbers in such way that there will be single array with
        # saw-like pattern but each saw tooth is slightly higher than the one before
        # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
        arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades,
                                                              number_of_distinct_colors_with_multiply_of_shades //
                                                              number_of_shades)

        # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly
        # higher than row above
        arr_by_shade_columns = arr_by_shade_rows.T

        # Keep number of saw teeth for later
        number_of_partitions = arr_by_shade_columns.shape[0]

        # Flatten the above matrix - join each row into single array
        nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

        # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
        initial_cm = hsv(nums_distributed_like_rising_saw)

        lower_partitions_half = number_of_partitions // 2
        upper_partitions_half = number_of_partitions - lower_partitions_half

        # Modify lower half in such way that colours towards beginning of partition are darker
        # First colours are affected more, colours closer to the middle are affected less
        lower_half = lower_partitions_half * number_of_shades
        for i in range(3):
            initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

        # Modify second half in such way that colours towards end of partition are less intense and brighter
        # Colours closer to the middle are affected less, colours closer to the end are affected more
        for i in range(3):
            for j in range(upper_partitions_half):
                modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) *
                                                                                                     number_of_shades, i]
                modifier = j * modifier / upper_partitions_half
                initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

        return ListedColormap(initial_cm)
