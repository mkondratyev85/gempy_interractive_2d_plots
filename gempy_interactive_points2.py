import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout, QCheckBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from math import cos,  sin, radians


class MplCanvas(FigureCanvasQTAgg):
    cell_number = 0
    direction = 'x'
    series_n = 0
    show_topography=True

    def update_constants(self):
        model_resolution = self.model.grid.regular_grid.resolution
        model_extension = self.model.grid.regular_grid.extent
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = model_extension
        self.x_res, self.y_res, self.z_res = model_resolution
        self.dx = (self.x_max - self.x_min) / self.x_res
        self.dy = (self.y_max - self.y_min) / self.y_res
        self.dz = (self.z_max - self.z_min) / self.z_res

    def __init__(self, model, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.model = model
        self.cmap = mcolors.ListedColormap(list(self.model._surfaces.df['color']))
        self.norm = mcolors.Normalize(vmin=0.5, vmax=len(self.cmap.colors) + 0.5)
        self.axes = self.fig.add_subplot(111)
        self.update_constants()
        plt.ion()

    def plot(self, update_image=True):
        if update_image:
            self.update_image()

        # _a, _b, _c, extent_val, x, y = self._slice(self.direction, self.cell_number)[:-2]

        self.axes.cla()
        self.axes.imshow(self.lith_image, origin='lower', zorder=-100, cmap=self.cmap,
                norm = self.norm, extent=self.extent_val,
                )
        self.plot_contact()
        if self.show_topography:
            self.plot_topography()
        self.fig.canvas.draw()
        self.update_plane_widget()

    def prepare_for_contacts(self):
        self._slice_ = self._slice(self.direction, self.cell_number)[:3]
        self.shape = self.model._grid.regular_grid.resolution
        self.scalar_fields = self.model.solutions.scalar_field_matrix

    def plot_contact(self):
        self.prepare_for_contacts()
        c_id = 0
        zorder = 100
        for e, block in enumerate(self.scalar_fields):
            level = self.model.solutions.scalar_field_at_surface_points[e][np.where(
                self.model.solutions.scalar_field_at_surface_points[e] != 0)]
            c_id2 = c_id + len(level)

            color_list = self.model._surfaces.df.groupby('isActive').get_group(True)['color'][c_id:c_id2][::-1]

            if self._slice_:
                image = block.reshape(self.shape)[self._slice_].T
            else:
                image = block.reshape(self.shape).T

            self.axes.contour(image, 0, levels=np.sort(level),
                       colors=color_list,
                       linestyles='solid', origin='lower',
                       extent=self.extent_val, 
                       zorder=zorder - (e + len(level))
                       )
            c_id = c_id2

    def calculate_p1p2(self, direction, cell_number):
        if direction == 'y':
            cell_number = int(self.model._grid.regular_grid.resolution[1] / 2) if cell_number == 'mid' else cell_number

            y = self.model._grid.regular_grid.extent[2] + self.model._grid.regular_grid.dy * cell_number
            p1 = [self.model._grid.regular_grid.extent[0], y]
            p2 = [self.model._grid.regular_grid.extent[1], y]

        elif direction == 'x':
            cell_number = int(self.model._grid.regular_grid.resolution[0] / 2) if cell_number == 'mid' else cell_number

            x = self.model._grid.regular_grid.extent[0] + self.model._grid.regular_grid.dx * cell_number
            p1 = [x, self.model._grid.regular_grid.extent[2]]
            p2 = [x, self.model._grid.regular_grid.extent[3]]
        return p1, p2
    
    def _slice_topo_4_sections(self, p1, p2, resx, method='interp2d'):
        """
        Slices topography along a set linear section
        Args:
            :param p1: starting point (x,y) of the section
            :param p2: end point (x,y) of the section
            :param resx: resolution of the defined section
            :param method: interpolation method, 'interp2d' for cubic scipy.interpolate.interp2d
                                             'spline' for scipy.interpolate.RectBivariateSpline
        Returns:
            :return: returns x,y,z values of the topography along the section
        """
        xy = self.model._grid.sections.calculate_line_coordinates_2points(p1, p2, resx)
        z = self.model._grid.sections.interpolate_zvals_at_xy(xy, self.model._grid.topography, method)
        return xy[:, 0], xy[:, 1], z


    def plot_topography(self):
        if self.direction == 'z':
            return
        p1, p2 = self.calculate_p1p2(self.direction, self.cell_number)
        resx = self.model._grid.regular_grid.resolution[0]
        resy = self.model._grid.regular_grid.resolution[1]

        x, y, z = self._slice_topo_4_sections(p1, p2, resx)
        if self.direction == 'x':
            a = np.vstack((y, z)).T
            ext = self.model._grid.regular_grid.extent[[2, 3]]
        elif self.direction == 'y':
            a = np.vstack((x, z)).T
            ext = self.model._grid.regular_grid.extent[[0, 1]]
        else:
            raise NotImplementedError
        a = np.append(a,
                      ([ext[1], a[:, 1][-1]],
                       [ext[1], self.model._grid.regular_grid.extent[5]],
                       [ext[0], self.model._grid.regular_grid.extent[5]],
                       [ext[0], a[:, 1][0]]))
        line = a.reshape(-1, 2)
        self.axes.fill(line[:, 0], line[:, 1], color='k')
    
    def update_image(self):
        lith_block = self.model.solutions.lith_block
        scalar_block = self.model.solutions.scalar_field_matrix[self.series_n]
        _a, _b, _c, self.extent_val, x, y = self._slice(self.direction, self.cell_number)[:-2]

        def reshape_block(block):
            plot_block = block.reshape(self.model._grid.regular_grid.resolution)
            return plot_block[_a, _b, _c].T

        self.lith_image = reshape_block(lith_block)
        self.scalar_image = reshape_block(scalar_block)

    def update_plane_widget(self):
            px, py, pz = self.plane_widget.GetOrigin()
            if self.direction == "x":
                px = self.get_coordinate()
            elif self.direction == "y":
                py = self.get_coordinate()
            elif self.direction == "z":
                pz = self.get_coordinate()
            self.plane_widget.SetOrigin(px, py, pz)
    
    def get_coordinate(self):
        if self.direction=='x':
            x = self.x_min + self.cell_number * self.dx
            return x
        elif self.direction=="y":
            y = self.y_min + self.cell_number * self.dy
            return y
        elif self.direction=="z":
            z = self.z_min + self.cell_number * self.dz
            return z


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

    def _slice(self, direction, cell_number):
        """
        Slice the 3D array (blocks or scalar field) in the specific direction selected in the plot functions
        """
        _a, _b, _c = (slice(0, self.model._grid.regular_grid.resolution[0]),
                      slice(0, self.model._grid.regular_grid.resolution[1]),
                      slice(0, self.model._grid.regular_grid.resolution[2]))

        if direction == "x":
            cell_number = int(self.model._grid.regular_grid.resolution[0] / 2) if cell_number == 'mid' else cell_number
            _a, x, y, Gx, Gy = cell_number, "Y", "Z", "G_y", "G_z"
            extent_val = self.model._grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == "y":
            cell_number = int(self.model._grid.regular_grid.resolution[1] / 2) if cell_number == 'mid' else cell_number
            _b, x, y, Gx, Gy = cell_number, "X", "Z", "G_x", "G_z"
            extent_val = self.model._grid.regular_grid.extent[[0, 1, 4, 5]]
        elif direction == "z":
            cell_number = int(self.model._grid.regular_grid.resolution[2] / 2) if cell_number == 'mid' else cell_number
            _c, x, y, Gx, Gy = cell_number, "X", "Y", "G_x", "G_y"
            extent_val = self.model._grid.regular_grid.extent[[0, 1, 2, 3]]
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

class CanvasX(MplCanvas):
    direction = 'x'

class CanvasY(MplCanvas):
    direction = 'y'

class CanvasZ(MplCanvas):
    direction = 'z'

class SectionCanvas(MplCanvas):
    # show_topography=False

    def __init__(self, section_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.section_name = section_name
        
    def update_plane_widget(self):
        pass

    def plot_topography(self):
        p1 = self.model._grid.sections.df.loc[self.section_name, 'start']
        p2 = self.model._grid.sections.df.loc[self.section_name, 'stop']
        x, y, z = self._slice_topo_4_sections(p1, p2, self.model._grid.topography.resolution[0])

        pseudo_x = np.linspace(0, self.model._grid.sections.df.loc[self.section_name, 'dist'], z.shape[0])
        a = np.vstack((pseudo_x, z)).T
        xy = np.append(a,
                       ([self.model._grid.sections.df.loc[self.section_name, 'dist'], a[:, 1][-1]],
                        [self.model._grid.sections.df.loc[self.section_name, 'dist'],
                         self.model._grid.regular_grid.extent[5]],
                        [0, self.model._grid.regular_grid.extent[5]],
                        [0, a[:, 1][0]])).reshape(-1, 2)

        self.axes.fill(xy[:, 0], xy[:, 1], 'k', zorder=10)

    def update_image(self):
        dist = self.model._grid.sections.df.loc[self.section_name, 'dist']
        self.extent_val = [0, dist, self.model._grid.regular_grid.extent[4], self.model._grid.regular_grid.extent[5]]

        l0, l1 = self.model._grid.sections.get_section_args(self.section_name)
        shape = self.model._grid.sections.df.loc[self.section_name, 'resolution']
        self.lith_image = self.model.solutions.sections[0][0][l0:l1].reshape(shape[0], shape[1]).T

    def prepare_for_contacts(self):
        self._slice_ = None
        l0, l1 = self.model._grid.sections.get_section_args(self.section_name)
        self.shape = self.model._grid.sections.df.loc[self.section_name, 'resolution']
        self.scalar_fields = self.model.solutions.sections[1][:, l0:l1]

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, gp, geo_model):
        super(MainWindow, self).__init__()
        self.gp = gp
        self.model = geo_model
        self.gui_model = GUIModel(geo_model)

        self.table_widget = MyTableWidget(self, self.model, self.gp, self.gui_model)
        self.setCentralWidget(self.table_widget)

        self.show()

    def closeEvent(self, event):
        self.table_widget.p3d.p.close()
        event.accept()

class GUIModel:

    def __init__(self, geo_model):
        self.geo_model = geo_model
        self.resolution = geo_model.grid.regular_grid.resolution
        self.extension = geo_model.grid.regular_grid.extent
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = self.extension
        self.x_res, self.y_res, self.z_res = self.resolution
        self.dx = (self.x_max - self.x_min) / self.x_res
        self.dy = (self.y_max - self.y_min) / self.y_res
        self.dz = (self.z_max - self.z_min) / self.z_res

    def list_of_sections(self):
        return self.geo_model._grid.sections.df.index.to_list()

class MyTableWidget(QWidget):
    
    def __init__(self, parent, model, gp, gui_model):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        self.model = model
        self.gp = gp
        self.gui_model = gui_model
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab_x = QWidget()
        self.tab_y = QWidget()
        self.tab_z = QWidget()
        self.tab_3d = QWidget()
        self.tabs.resize(300,200)

        
        # Add tabs
        self.tabs.addTab(self.tab_x, "X")
        self.tabs.addTab(self.tab_y, "Y")
        self.tabs.addTab(self.tab_z, "Z")
        # self.tabs.addTab(self.tab_3d, "3D")



        self.canvas_x = CanvasX(model, self.tab_x, width=5, height=4, dpi=100)
        self.canvas_y = CanvasY(model, self.tab_y, width=5, height=4, dpi=100)
        self.canvas_z = CanvasZ(model, self.tab_z, width=5, height=4, dpi=100)
        
        # Create first tab
        self.tab_x.layout = QVBoxLayout(self.tabs)
        self.tab_x.layout.addWidget(self.canvas_x)
        self.tab_x.setLayout(self.tab_x.layout)

        self.tab_y.layout = QVBoxLayout(self.tabs)
        self.tab_y.layout.addWidget(self.canvas_y)
        self.tab_y.setLayout(self.tab_y.layout)
        
        self.tab_z.layout = QVBoxLayout(self.tabs)
        self.tab_z.layout.addWidget(self.canvas_z)
        self.tab_z.setLayout(self.tab_z.layout)

        self.tabs_section = []
        self.section_canvases = []
        for section in self.gui_model.list_of_sections():
            tab = QWidget()
            self.tabs_section.append(tab)
            self.tabs.addTab(tab, section)

            canvas = SectionCanvas(model=model, section_name=section, parent=tab, width=5, height=4, dpi=100)
            
            tab.layout = QVBoxLayout(self.tabs)
            tab.layout.addWidget(canvas)
            tab.setLayout(tab.layout)
            self.section_canvases.append(canvas)


        self.p3d = gp.plot_3d(self.model, plotter_type='background',notebook=False,show_lith=False,show_surfaces=True,show_topography=True,live_updating=False,show_results=True,show_data=False,)

        def update_func(normal, origin):
            pass

        self.p3d.p.clear_plane_widgets()
        self.canvas_x.plane_widget = self.p3d.p.add_plane_widget(update_func, test_callback=False, normal_rotation=False)
        self.canvas_x.plane_widget.SetNormal(1.0, 0.0, 0.0)
        self.canvas_y.plane_widget = self.p3d.p.add_plane_widget(update_func, test_callback=False, normal_rotation=False)
        self.canvas_y.plane_widget.SetNormal(0.0, 1.0, 0.0)
        self.canvas_z.plane_widget = self.p3d.p.add_plane_widget(update_func, test_callback=False, normal_rotation=False)
        self.canvas_z.plane_widget.SetNormal(0.0, 0.0, 1.0)

        self.layout.addWidget(self.tabs)
        self.button_prev = QPushButton(self)
        self.button_prev.setText("<")
        self.button_prev.clicked.connect(self.prev_on_click)

        self.button_next = QPushButton(self)
        self.button_next.setText(">")
        self.button_next.clicked.connect(self.next_on_click)

        self.layout.addWidget(self.button_prev)
        self.layout.addWidget(self.button_next)
        
        self.cbtn_show_relief = QCheckBox(self)
        self.cbtn_show_relief.setText("Show relief")
        self.cbtn_show_relief.stateChanged.connect(self.show_relief_changed)
        self.layout.addWidget(self.cbtn_show_relief)

        self.setLayout(self.layout)

        has_topography = self.model._grid.active_grids[2]
        self.cbtn_show_relief.setEnabled(has_topography)

        self.canvases = [self.canvas_x, self.canvas_y, self.canvas_z]
        for canvas in self.canvases + self.section_canvases:
            canvas.plot()

        self.tabs.currentChanged.connect(self.on_change)

    def show_relief_changed(self):
        is_show_relief = self.cbtn_show_relief.isChecked()
        for canvas in self.canvases + self.section_canvases:
            canvas.show_topography = is_show_relief
            canvas.plot()

    def on_change(self):
        is_enabled = self.tabs.currentIndex() < 3
        self.button_next.setEnabled(is_enabled)
        self.button_prev.setEnabled(is_enabled)

    def prev_on_click(self):
        canvas = self.canvases[self.tabs.currentIndex()]
        canvas.switch_section("down")

    def next_on_click(self):
        canvas = self.canvases[self.tabs.currentIndex()]
        canvas.switch_section("up")
        

def start_app(gp, geo_model):
    app = QtWidgets.QApplication(["a"])
    w = MainWindow(gp, geo_model)
    app.exec()

print("aaaa")






# app = QtWidgets.QApplication(["a"])
# w = MainWindow()
# app.exec()





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
