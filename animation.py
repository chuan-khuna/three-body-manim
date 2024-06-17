from manim import *

config["frame_size"] = (1440, 1440)
config["frame_height"] = 3
config["frame_width"] = 3
FPS = 60
config["frame_rate"] = FPS
config["background_color"] = BLACK
config['disable_caching_warning'] = True


import numpy as np

from utils.vis_utils import *

from libs.body import Body
from libs.constants import *

from copy import deepcopy
import pandas as pd

from tqdm import tqdm


def initialise_stars(configs, masses, G=1):
    # https://observablehq.com/@rreusser/periodic-planar-three-body-orbits
    stars = {}
    num_stars = len(masses)
    for i in range(num_stars):
        idx = 4 * i
        position = [configs[idx], configs[idx + 1]]
        velocity = [configs[idx + 2], configs[idx + 3]]
        weight = masses[i]
        star = Body(weight, np.array(position), np.array(velocity), f'star_{i}')
        star.G = G
        stars[f'star_{i}'] = star
    return stars


colours = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE, TEAL, PINK, GREY]


class NBodyAnimation(Scene):

    def __init__(self):
        super().__init__()

        position_velocity = [
            -1,
            0,
            0.339393,
            0.536191,
            1,
            0,
            0.339393,
            0.536191,
            0,
            0,
            -0.678786,
            -1.072382,
        ]
        masses = [1, 1, 1]

        self.stars = initialise_stars(configs=position_velocity, masses=masses, G=1)

        self.dt = 1e-4

        # how long to run the simulation for
        # how many it will repeat/completed
        self.num_steps = 200e3
        self.df = self.compute()

        self.star_radius = 0.025
        self.trace_length = 40

    def compute(self) -> pd.DataFrame:
        star_positions = {}

        stars = deepcopy(self.stars)

        for k, star in stars.items():
            star_positions[k] = [star.position]

        for step in tqdm(
            range(0, int(self.num_steps)),
            desc='Computing trajectories',
            ncols=100,
            total=int(self.num_steps),
        ):
            # current states
            backup_stars = deepcopy(stars)

            for k, star in stars.items():
                other_stars = [other for k, other in backup_stars.items()]
                updated_star = star.update(other_stars, self.dt)
                stars[k] = updated_star
                star_positions[k].append(updated_star.position)

        dfs = []
        for k, v in star_positions.items():
            df = pd.DataFrame(v, columns=['x', 'y'])
            df['star'] = k
            dfs.append(df)

        df = pd.concat(dfs).reset_index(drop=True)

        fig = plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df, x='x', y='y', hue='star', s=2)
        ax = plt.gca()
        ax.legend().set_visible(False)
        plt.savefig('nbody.png')

        return df

    def _sample_df(self, df, num_samples=1000):
        """To reduce the number of points in the plot
        sample data from the calculated trajectories
        """
        star_names = df['star'].unique()
        sampled_dfs = []
        for star_name in star_names:
            star_df = df[df['star'] == star_name].reset_index(drop=True)
            step_size = len(star_df) // num_samples
            sampled_df = star_df.iloc[::step_size]
            sampled_dfs.append(sampled_df)
        return pd.concat(sampled_dfs).reset_index(drop=True)

    def create_star(self, x, y, colour):
        # print(f"Creating star at {x}, {y}")
        star_circle = Circle(radius=self.star_radius, color=colour, fill_opacity=1)
        star_circle.move_to([x, y, 0])
        return star_circle

    def _get_star_df_dict(self, df):
        star_dict = {}
        for star_name in df['star'].unique():
            star_df = df[df['star'] == star_name].reset_index(drop=True)
            star_dict[star_name] = star_df
        return star_dict

    def construct(self):

        # how long the output will be
        # scaling simulation steps
        num_samples = 500
        print(f"Number of samples: {num_samples}, at {FPS} fps = {num_samples / FPS} seconds")

        # how long the trace will be
        # percent of the total loop (simulated steps)
        self.trace_length = int(num_samples * 0.5)

        frame_time = 0.001

        STROKE_WIDTH_START = 2.0
        OPACITY_START = 1.0

        df = self._sample_df(self.df, num_samples=num_samples)
        df_by_star = self._get_star_df_dict(df)

        star_names = df['star'].unique()

        def trace_updater(traceobj, starobj):
            color = starobj.get_color()
            last_line = traceobj[-1]
            new_line = Line(
                last_line.get_end(),
                starobj.get_center(),
                color=color,
                stroke_width=STROKE_WIDTH_START,
                stroke_opacity=OPACITY_START,
            )
            traceobj.add(new_line)

            if len(traceobj) > self.trace_length:
                traceobj.remove(traceobj[0])

            # update opacity and stroke width
            opacity_decay = 0.99
            stroke_decay = 0.99
            for i, line in enumerate(traceobj[::-1]):
                opacity = max(0.25, OPACITY_START * opacity_decay**i)
                stroke_width = max(0.1, STROKE_WIDTH_START * stroke_decay**i)
                line.set_stroke(color=line.get_stroke_color(), width=stroke_width, opacity=opacity)

        star_objs = {}
        star_trackers: dict[str, list['ValueTracker']] = {}
        star_traces = {}

        # intialise stars
        for i, star_name in enumerate(star_names):
            star_df = df_by_star[star_name]
            star = self.create_star(star_df['x'].iloc[0], star_df['y'].iloc[0], colours[i])

            x_tracker = ValueTracker(df_by_star[star_name]['x'].iloc[0])
            y_tracker = ValueTracker(df_by_star[star_name]['y'].iloc[0])
            star_trackers[star_name] = [x_tracker, y_tracker]
            star.add_updater(
                lambda starobj, starname=star_name: starobj.move_to(
                    [
                        star_trackers[starname][0].get_value(),
                        star_trackers[starname][1].get_value(),
                        0,
                    ]
                )
            )

            # initialise trace
            trace = VGroup()
            trace.add(Line(star.get_center(), star.get_center(), color=star.get_color()))

            # trace updater, depends on star position
            trace.add_updater(lambda traceobj, starobj=star: trace_updater(traceobj, starobj))

            star_objs[star_name] = star
            star_traces[star_name] = trace

        # initialise starting conditions
        self.add(*[star for star in star_objs.values()])
        self.add(*[trace for trace in star_traces.values()])
        self.wait(0.25)

        # animate
        for step in range(num_samples):
            args = []
            new_traces = {}
            for star_name in star_names:
                # update star positions
                x_tracker, y_tracker = star_trackers[star_name]
                x = df_by_star[star_name]['x'].iloc[step]
                y = df_by_star[star_name]['y'].iloc[step]
                args.append((x_tracker, x))
                args.append((y_tracker, y))

            # create animations for each star
            star_animations = [tracker.animate.set_value(val) for tracker, val in args]

            self.play(*(star_animations), run_time=frame_time)

            star_traces = new_traces
