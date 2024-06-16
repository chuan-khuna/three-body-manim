from manim import *

config["frame_size"] = (2560, 1080)
config["frame_rate"] = 60
config["background_color"] = BLACK

import numpy as np

from utils.vis_utils import *

from libs.body import Body
from libs.constants import *

from copy import deepcopy
import pandas as pd

from tqdm import tqdm


def initialise_stars(configs, weights, G=1):
    # https://observablehq.com/@rreusser/periodic-planar-three-body-orbits
    stars = {}
    num_stars = len(weights)
    for i in range(num_stars):
        idx = 4 * i
        position = [configs[idx], configs[idx + 1]]
        velocity = [configs[idx + 2], configs[idx + 3]]
        weight = weights[i]
        star = Body(weight, np.array(position), np.array(velocity), f'star_{i}')
        star.G = G
        stars[f'star_{i}'] = star
    return stars


colours = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE, TEAL, PINK, GREY]


class NBodyAnimation(Scene):

    def __init__(self):
        super().__init__()
        self.stars = initialise_stars(
            configs=[
                -0.337076702,
                0,
                0,
                0.9174260238,
                2.1164029743,
                0,
                0,
                -0.0922665014,
                -1.7793262723,
                0,
                0,
                -0.8251595224,
            ],
            weights=[1, 1, 1],
            G=1,
        )

        # self.stars = initialise_stars(
        #     configs=[
        #         -0.7283341038,
        #         0,
        #         0,
        #         0.8475982451,
        #         2.8989177778,
        #         0,
        #         0,
        #         -0.0255162097,
        #         -2.1705836741,
        #         0,
        #         0,
        #         -0.8220820354,
        #     ],
        #     weights=[1, 1, 1],
        #     G=1,
        # )

        self.dt = 1e-3
        self.num_steps = 100_000

        self.star_radius = 0.025

        self.df = self.compute()

    def compute(self):
        star_positions = {}

        stars = deepcopy(self.stars)

        for k, star in stars.items():
            star_positions[k] = [star.position]

        for i in tqdm(
            range(0, int(self.num_steps)),
            desc='Computing trajectories',
            ncols=100,
            total=int(self.num_steps),
        ):

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
        sns.scatterplot(data=df, x='x', y='y', hue='star', s=4)
        plt.savefig('nbody.png')
        return df

    def sampling_df(self, df, num_samples=1000):
        star_names = df['star'].unique()
        sampled_dfs = []
        for star_name in star_names:
            star_df = df[df['star'] == star_name].reset_index(drop=True)
            step_size = len(star_df) // num_samples
            sampled_df = star_df.iloc[::step_size]
            sampled_dfs.append(sampled_df)
        return pd.concat(sampled_dfs).reset_index(drop=True)

    def construct(self):
        num_samples = 500
        move_run_time = 0.001

        df = self.sampling_df(self.df, num_samples=num_samples)

        star_names = df['star'].unique()

        star_objs = {}
        star_trace_objs = {}
        star_df_dict = {
            star_name: df[df['star'] == star_name].reset_index(drop=True)
            for star_name in star_names
        }
        star_full_df_dict = {
            star_name: df[df['star'] == star_name].reset_index(drop=True)
            for star_name in star_names
        }

        # intialise stars
        for i, star_name in enumerate(star_names):
            star_df = df[df['star'] == star_name].reset_index(drop=True)
            star = Circle(radius=self.star_radius, color=colours[i], fill_opacity=1).shift(
                [star_df['x'].iloc[0], star_df['y'].iloc[0], 0]
            )
            star_objs[star_name] = star

        for star_obj in star_objs.values():
            self.add(star_obj)

        # self.play(*[Write(star_obj) for star_obj in star_objs.values()])
        self.wait(0.25)

        for i in range(num_samples):
            new_star_objs = {}
            new_star_trace_objs = {}
            transformations = []
            for star_name in star_names:
                # current star object
                star = star_objs[star_name]

                # trace
                max_trace_length = 10
                start_idx = max(0, i - max_trace_length)
                if star_name not in star_trace_objs.keys():
                    star_trace_objs[star_name] = self.create_trace(
                        star,
                        star_df_dict[star_name]['x'].iloc[start_idx:i],
                        star_df_dict[star_name]['y'].iloc[start_idx:i],
                    )
                    self.add(star_trace_objs[star_name])
                else:
                    new_trace = self.create_trace(
                        star,
                        star_df_dict[star_name]['x'].iloc[start_idx:i],
                        star_df_dict[star_name]['y'].iloc[start_idx:i],
                    )
                    transformations.append(
                        ReplacementTransform(star_trace_objs[star_name], new_trace)
                    )

                # update position
                newx = star_df_dict[star_name]['x'].iloc[i]
                newy = star_df_dict[star_name]['y'].iloc[i]
                moved_star = self.move_star(star, newx, newy)

                # prepare for next iteration
                new_star_objs[star_name] = moved_star
                transformations.append(ReplacementTransform(star, moved_star))

            self.play(*transformations, run_time=move_run_time)
            star_objs = new_star_objs
            star_trace_objs = new_star_trace_objs

    def move_star(self, star_obj, x, y):
        return star_obj.copy().move_to([x, y, 0])

    def create_trace(self, star_obj, xs, ys):
        color = star_obj.get_color()
        xs = np.array(xs)
        ys = np.array(ys)
        dots = [Dot([xs[i], ys[i], 0], color=color, radius=0.005) for i in range(len(xs))]
        arcs = [
            Line(color=color, stroke_width=0.2, stroke_opacity=0.25).put_start_and_end_on(
                dots[i].get_center(), dots[i + 1].get_center()
            )
            for i in range(len(dots) - 1)
        ]
        return VGroup(*arcs)
