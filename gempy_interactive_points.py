from math import cos,  sin, radians
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

path_to_img = "/tmp/backim.png"

class GempyInteract:

    direction = 'x'
    cell_number=5
    angle_of_rotation = radians(5)
    scalar_index = 0
    mode = "add"
    formation_to_add = None
    show_topography = False
    show_scalar = False
    show_lith = False
    show_img = False
    image_shown = None
    images = []

    def __init__(self, gp, geo_model, p3d=None, surfaces_for_3d=None,
            images=None):
        self.gp = gp
        self.geo_model = geo_model
        model_resolution = geo_model.grid.regular_grid.resolution
        model_extension = geo_model.grid.regular_grid.extent
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = model_extension
        self.x_res, self.y_res, self.z_res = model_resolution
        self.dx = (self.x_max - self.x_min) / self.x_res
        self.dy = (self.y_max - self.y_min) / self.y_res
        self.dz = (self.z_max - self.z_min) / self.z_res
        self.p3d = p3d
        self.surfaces_for_3d = surfaces_for_3d
        self.initial_plot()
        self.connect_events()
        self.create_colors()
        self.add_buttons()
        self.plot()
        if images:
            for image_path, extent, direction, origin_coordinate in images:
                image = mpimg.imread(image_path)
                self.images.append((image, extent, direction, origin_coordinate))

    def add_buttons(self):
        fig = self.axis.get_figure()
        self.ax_next = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.ax_prev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.ax_working_with = fig.add_axes([0.59, 0.05, 0.1, 0.125])
        self.ax_surfaces = fig.add_axes([0.47, 0.05, 0.1, 0.125])
        self.ax_directions = fig.add_axes([0.33, 0.05, 0.1, 0.125])
        self.ax_delete = fig.add_axes([0.20, 0.05, 0.1, 0.125])
        self.ax_settings = fig.add_axes([0.00, 0.05, 0.1, 0.125])
        self.next_button = Button(self.ax_next, 'Next')
        self.next_button.on_clicked(self.next_section)
        self.prev_button = Button(self.ax_prev, 'Prev')
        self.prev_button.on_clicked(self.prev_section)
        self.delete_button = Button(self.ax_delete, 'Delete')
        self.delete_button.on_clicked(self.delete_point)
        self.working_with_rbtn = RadioButtons(self.ax_working_with, ('Points', 'Orientations'))
        self.working_with_rbtn.on_clicked(self.change_working_with)
        self.change_working_with('Points')
        self.direction_rbtn = RadioButtons(self.ax_directions, ('x','y','z'))
        self.direction_rbtn.on_clicked(self.direction_changed)
        self.direction_changed('x')
        self.settings_button = CheckButtons(self.ax_settings,('show_topography', 'show_scalar', 'show_lith', 'show_img'))
        self.settings_button.on_clicked(self.settings_changed)

        surfaces = self.geo_model._surfaces.df.surface.to_list()[:-1]
        self.surfaces_rbtn = RadioButtons(self.ax_surfaces, surfaces)
        self.surfaces_rbtn.on_clicked(self.surface_changed)
        self.surface_changed(surfaces[0])

    def settings_changed(self, label):
        if label == 'show_topography':
            self.show_topography = not self.show_topography
        if label == 'show_scalar':
            self.show_scalar = not self.show_scalar
        if label == 'show_lith':
            self.show_lith = not self.show_lith
        if label == 'show_img':
            self.show_img = not self.show_img
        self.plot()

    def delete_point(self, event):
        self.mode = "delete"

    def direction_changed(self, label):
        self.direction = label
        self.axis = self.p2d.add_section(cell_number=self.cell_number, direction=self.direction, show_topography=False)
        self.plot()

    def surface_changed(self, label):
        self.formation_to_add = label
        series = self.geo_model._surfaces.df[self.geo_model._surfaces.df.surface == label].series.values[0]
        self.scalar_index = self.geo_model._series.df[self.geo_model._series.df.index == series].order_series.values[0] - 1
        self.plot()

    def next_section(self, event):
        self.switch_section("up")

    def prev_section(self, event):
        self.switch_section("down")

    def change_working_with(self, label):
        if label == 'Points':
            self.working_with = 'points'
        elif label == 'Orientations':
            self.working_with = 'orientations'
    
    def create_colors(self):
        self.colors = dict()
        for i, row in self.geo_model._surfaces.df.iterrows():
            self.colors[row.surface] = row.color

    def initial_plot(self):
        self.p2d = self.gp.plot_2d(self.geo_model, n_axis=3, section_names=None, direction=None, cell_number=None)
        self.axis = self.p2d.add_section(cell_number=self.cell_number, direction=self.direction, show_topography=False)

        if self.p3d:
            def update_func(normal, origin):
                pass

            self.p3d.p.clear_plane_widgets()
            self.plane_widget = self.p3d.p.add_plane_widget(update_func, test_callback=False, normal_rotation=False)

        plt.ion()

    def switch_section(self, where):
        if where == "up":
            self.cell_number += 1
        if where == "down":
            self.cell_number -= 1
        if self.cell_number <0:
            self.cell_number =0
        if self.direction=='x':
            if self.cell_number > self.x_res-1:
                self.cell_number = self.x_res-1
        elif self.direction=="y":
            if self.cell_number > self.y_res-1:
                self.cell_number = self.y_res-1
        elif self.direction=="z":
            if self.cell_number > self.z_res-1:
                self.cell_number = self.z_res-1
        self.plot()

    def onkey_release(self, event):
        pass

    def delete_on_click(self, event):
        self.mode="add"
        plot_x, plot_y = event.xdata, event.ydata
        if self.working_with == "points":
            df = self.geo_model._surface_points.df
        else:
            df = self.geo_model._orientations.df

        if plot_x is None or plot_y is None:
            return
        if self.direction=='x':
            x = self.x_min + self.cell_number * self.dx
            mask = abs(df.X - x) < .5*self.dx
            elements = df[mask]
            elements_plot_x = elements.Y
            elements_plot_y = elements.Z
        elif self.direction=='y':
            y = self.y_min + self.cell_number * self.dy
            # orientations = self.geo_model._orientations.df[mask]
            # orientations_plot_x = orientations.X
            # orientations_plot_y = orientations.Z

            mask = abs(df.Y - y) < .5*self.dy
            elements = df[mask]
            elements_plot_x = elements.X
            elements_plot_y = elements.Z
        elif self.direction=='z':
            z = self.z_min + self.cell_number * self.dz
            # mask = abs(self.geo_model._orientations.df.Z - z) < .5*self.dz
            # orientations = self.geo_model._orientations.df[mask]
            # orientations_plot_x = orientations.X
            # orientations_plot_y = orientations.Y
            mask = abs(df.Z - z) < .5*self.dz
            elements = df[mask]
            elements_plot_x = elements.X
            elements_plot_y = elements.Y
        distance_sq = (elements_plot_x - plot_x)**2 + (elements_plot_y - plot_y)**2
        try:
            closest_index = distance_sq.sort_values().index[0]
        except IndexError:
            return
        if self.working_with == "points":
            self.geo_model.delete_surface_points(closest_index)
        else:
            self.geo_model.delete_orientations(closest_index)
        self.gp.compute_model(self.geo_model)
        self.plot()

    def onclick(self, event):
        if self.mode == "delete":
            self.delete_on_click(event)
            return
        # y, z = event.xdata, event.ydata
        plot_x, plot_y = event.xdata, event.ydata
        if not event.inaxes == self.axis:
            return
        if plot_x is None or plot_y is None:
            return
        if self.direction == "x":
            x = self.x_min + self.cell_number * self.dx
            y = plot_x
            z = plot_y
        elif self.direction == "y":
            y = self.y_min + self.cell_number * self.dy
            x = plot_x
            z = plot_y
        elif self.direction == "z":
            z = self.z_min + self.cell_number * self.dz
            x = plot_x
            y = plot_y
        if self.working_with == 'points':
            self.add_point_at_coord(x, y, z, formation=self.formation_to_add)
        elif self.working_with == "orientations":
            pole_vector = (0.0, 0.0, 1.0)
            if self.direction == "z":
                pole_vector = (0.1, 0.0, 0.0)
            self.add_orientation_at_coord(x, y, z,formation=self.formation_to_add, pole_vector=pole_vector)
        self.gp.compute_model(self.geo_model)
        self.plot()

    def onscroll(self, event):
        plot_x, plot_y = event.xdata, event.ydata
        rotation_factor = 1 if event.button == "up" else -1

        
        if plot_x is None or plot_y is None:
            return
        if self.direction=='x':
            x = self.x_min + self.cell_number * self.dx
            mask = abs(self.geo_model._orientations.df.X - x) < .5*self.dx
            orientations = self.geo_model._orientations.df[mask]
            orientations_plot_x = orientations.Y
            orientations_plot_y = orientations.Z
        elif self.direction=='y':
            y = self.y_min + self.cell_number * self.dy
            mask = abs(self.geo_model._orientations.df.Y - y) < .5*self.dy
            orientations = self.geo_model._orientations.df[mask]
            orientations_plot_x = orientations.X
            orientations_plot_y = orientations.Z
        elif self.direction=='z':
            z = self.z_min + self.cell_number * self.dz
            mask = abs(self.geo_model._orientations.df.Z - z) < .5*self.dz
            orientations = self.geo_model._orientations.df[mask]
            orientations_plot_x = orientations.X
            orientations_plot_y = orientations.Y
        distance_sq = (orientations_plot_x - plot_x)**2 + (orientations_plot_y - plot_y)**2
        try:
            closest_index = distance_sq.sort_values().index[0]
        except IndexError:
            return
        x = self.geo_model._orientations.df.at[closest_index, 'X']
        y = self.geo_model._orientations.df.at[closest_index, 'Y']
        z = self.geo_model._orientations.df.at[closest_index, 'Z']
        surface = self.geo_model._orientations.df.at[closest_index, 'surface']
        g_x = self.geo_model._orientations.df.at[closest_index, 'G_x']
        g_y = self.geo_model._orientations.df.at[closest_index, 'G_y']
        g_z = self.geo_model._orientations.df.at[closest_index, 'G_z']
        old_x = self.geo_model._orientations.df.at[closest_index, 'X']
        old_y = self.geo_model._orientations.df.at[closest_index, 'Y']
        old_z = self.geo_model._orientations.df.at[closest_index, 'Z']

        if self.direction=='x':
            new_gy, new_gz = rotate_a_vector(g_y, g_z, rotation_factor * self.angle_of_rotation)
            new_gx = g_x
        elif self.direction=='y':
            new_gx, new_gz = rotate_a_vector(g_x, g_z, rotation_factor * self.angle_of_rotation)
            new_gy = g_y
        elif self.direction=='z':
            new_gx, new_gy = rotate_a_vector(g_x, g_y, rotation_factor * self.angle_of_rotation)
            new_gz = g_z

        
        # self.geo_model.delete_orientations(closest_index)
        # self.add_orientation_at_coord(x, y, z,formation=surface, pole_vector=(new_gx, new_gy, new_gz))
        self.geo_model.modify_orientations(idx=closest_index, G_x=new_gx, G_y=new_gy, G_z=new_gz)
        self.gp.compute_model(self.geo_model)
        self.plot()

    def connect_events(self):
        fig = self.axis.get_figure()
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        fig.canvas.mpl_connect('scroll_event', self.onscroll)
        fig.canvas.mpl_connect('key_release_event', self.onkey_release)

    def plot(self):
        ax = self.axis
        p2d = self.p2d
        cell_number = self.cell_number
        direction = self.direction
        geo_model = self.geo_model
        p2d.remove(ax)

        if self.show_lith:
            p2d.plot_lith(ax, cell_number=cell_number, direction=direction)
        try:
            p2d.plot_contacts(ax, cell_number=cell_number, direction=direction)
        except ValueError:
            pass
        if self.show_scalar:
            p2d.plot_scalar_field(ax, direction=direction, cell_number=cell_number, series_n=self.scalar_index)
        if direction=='x':
            x = self.x_min + cell_number * self.dx
            mask = abs(geo_model._surface_points.df.X - x) < .5*self.dx
            points = geo_model._surface_points.df[mask]
            points_plot_x = points.Y.to_list()
            points_plot_y = points.Z.to_list()
            mask = abs(geo_model._orientations.df.X - x) < .5*self.dx
            orientations = geo_model._orientations.df[mask]
            orientations_plot_x = orientations.Y.to_list()
            orientations_plot_dx = orientations.G_y.to_list()
            orientations_plot_y = orientations.Z.to_list()
            orientations_plot_dy = orientations.G_z.to_list()
            self.axis.set_title(f"x: {x}, cell_number: {cell_number}")
        elif direction=="y":
            y = self.y_min + cell_number * self.dy
            mask = abs(geo_model._surface_points.df.Y - y) < .5*self.dy
            points = geo_model._surface_points.df[mask]
            points_plot_x = points.X.to_list()
            points_plot_y = points.Z.to_list()
            mask = abs(geo_model._orientations.df.Y - y) < .5*self.dy
            orientations = geo_model._orientations.df[mask]
            orientations_plot_x = orientations.X.to_list()
            orientations_plot_dx = orientations.G_x.to_list()
            orientations_plot_y = orientations.Z.to_list()
            orientations_plot_dy = orientations.G_z.to_list()
            self.axis.set_title(f"y: {y}, cell_number: {cell_number}")
        elif direction=="z":
            z = self.z_min + cell_number * self.dz
            mask = abs(geo_model._surface_points.df.Z - z) < .5*self.dz
            points = geo_model._surface_points.df[mask]
            points_plot_x = points.X.to_list()
            points_plot_y = points.Y.to_list()
            mask = abs(geo_model._orientations.df.Z - z) < .5*self.dz
            orientations = geo_model._orientations.df[mask]
            orientations_plot_x = orientations.X.to_list()
            orientations_plot_y = orientations.Y.to_list()
            orientations_plot_dx = orientations.G_x.to_list()
            orientations_plot_dy = orientations.G_y.to_list()
            self.axis.set_title(f"z: {z}, cell_number: {cell_number}")

        # if self.formation_to_add:
        #     orientations = orientations[orientations.surface == self.formation_to_add]
        points_surfaces = points.surface.to_list()
        orientations_surfaces = orientations.surface.to_list()

        points_colors = [self.colors[surface] for surface in points_surfaces]
        orientations_colors = [self.colors[surface] for surface in orientations_surfaces]

        if self.show_topography:
            self.p2d.plot_topography(ax, cell_number=cell_number, direction=direction)
        ax.scatter(points_plot_x, points_plot_y, c=points_colors, zorder=103)
        ax.quiver(orientations_plot_x, orientations_plot_y, orientations_plot_dx, orientations_plot_dy, color=orientations_colors, zorder=103)

        self.axis.get_figure().canvas.draw()


        if self.p3d:
            px, py, pz = self.plane_widget.GetOrigin()
            if direction == "x":
                pnormal = (1.0, 0.0, 0.0)
                px = x
            elif direction == "y":
                pnormal = (0.0, 1.0, 0.0)
                py = y
            elif direction == "z":
                pnormal = (0.0, 0.0, 1.0)
                pz = z
            self.plane_widget.SetNormal(pnormal)
            self.plane_widget.SetOrigin(px, py, pz)

            if self.surfaces_for_3d:
                self.p3d.plot_surfaces(self.surfaces_for_3d, opacity=0.4)

        if self.image_shown:
            self.image_shown.remove()
            self.image_shown = None
        if self.show_img:
            if direction == 'x':
                self.plot_image(direction, x)
            if direction == 'y':
                self.plot_image(direction, y)
            if direction == 'z':
                self.plot_image(direction, z)

    def plot_image(self, direction_, coordinate):
        for image, extent, direction, origin_coordinate in self.images:
            if direction != direction_:
                continue
            d_coord = abs(coordinate - origin_coordinate)
            if direction == "x":
                if d_coord > self.dx:
                    continue
            if direction == "y":
                if d_coord > self.dy:
                    continue
            if direction == "z":
                if d_coord > self.dz:
                    continue

            self.image_shown = self.axis.imshow(image, origin='upper', alpha=.8, extent = extent)
            # print(i)
            # print(dir(i))

    def add_point_at_coord(self, x, y, z, formation):
        self.geo_model.add_surface_points(X=x, Y=y, Z=z, surface=formation)

    def add_orientation_at_coord(self, x, y, z, formation, pole_vector):
        self.geo_model.add_orientations(X=x, Y=y, Z=z, surface=formation, pole_vector=pole_vector)








def rotate_a_vector(x, y, angle):
    x_ = x * cos(angle) - y * sin(angle)
    y_ = x * sin(angle) + y * cos(angle)
    return x_, y_
