#! /usr/bin/env python3
import h5py as h5
import numpy as np

# MODEL GEOMETRY
# beam
span_main = 16.0
lambda_main = 0.25
lambda_dihedral = 20*np.pi/180
ea_main = 0.3

length_fuselage = 10
offset_fuselage = 1.25*0
sigma_fuselage = 10
m_bar_fuselage = 0.3
j_bar_fuselage = 0.1
ea = 1e4
ga = 1e4
gj = 1e4
eiy = 2e4
eiz = 70*eiy
# eiz = 4e6
m_bar_main = 0.75 + 3.125 / 2
j_bar_main = 0.4

span_tail = 2.5
ea_tail = 0.5
fin_height = 2.5
ea_fin = 0.5
sigma_tail = 10
m_bar_tail = 0.3
j_bar_tail = 0.1


class HaleStructure:

    def __init__(self, case_name, case_route, **kwargs):

        self.sigma = kwargs.get('sigma', 1)
        self.n_elem_multiplier = kwargs.get('n_elem_multiplier', 1.5)

        self.route = case_route
        self.case_name = case_name

        self.thrust = kwargs.get('thrust', 0.)

        self.n_elem = None
        self.n_node = None
        self.n_node_elem = 3

        self.x = None
        self.y = None
        self.z = None

        self.n_elem_main = None
        self.n_elem_fuselage = None
        self.n_elem_fin = None
        self.n_elem_tail = None

        self.n_node_main = None
        self.n_node_fuselage = None
        self.n_node_fin = None
        self.n_node_tail = None

    def set_thrust(self, value):
        self.thrust = value

    def generate(self):

        n_elem_multiplier = self.n_elem_multiplier
        sigma = self.sigma

        n_elem_main = int(4*n_elem_multiplier)
        n_elem_tail = int(2*n_elem_multiplier)
        n_elem_fin = int(2*n_elem_multiplier)
        n_elem_fuselage = int(2*n_elem_multiplier)

        # lumped masses
        n_lumped_mass = 1
        lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
        lumped_mass = np.zeros((n_lumped_mass, ))
        lumped_mass[0] = 5
        lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
        lumped_mass_position = np.zeros((n_lumped_mass, 3))


        # beam processing
        n_node_elem = self.n_node_elem
        span_main1 = (1.0 - lambda_main)*span_main
        span_main2 = lambda_main*span_main

        n_elem_main1 = round(n_elem_main*(1 - lambda_main))
        n_elem_main2 = n_elem_main - n_elem_main1

        # total number of elements
        n_elem = 0
        n_elem += n_elem_main1 + n_elem_main1
        n_elem += n_elem_main2 + n_elem_main2
        n_elem += n_elem_fuselage
        n_elem += n_elem_fin
        n_elem += n_elem_tail + n_elem_tail

        # number of nodes per part
        n_node_main1 = n_elem_main1*(n_node_elem - 1) + 1
        n_node_main2 = n_elem_main2*(n_node_elem - 1) + 1
        n_node_main = n_node_main1 + n_node_main2 - 1
        n_node_fuselage = n_elem_fuselage*(n_node_elem - 1) + 1
        n_node_fin = n_elem_fin*(n_node_elem - 1) + 1
        n_node_tail = n_elem_tail*(n_node_elem - 1) + 1

        # total number of nodes
        n_node = 0
        n_node += n_node_main1 + n_node_main1 - 1
        n_node += n_node_main2 - 1 + n_node_main2 - 1
        n_node += n_node_fuselage - 1
        n_node += n_node_fin - 1
        n_node += n_node_tail - 1
        n_node += n_node_tail - 1


        # stiffness and mass matrices
        n_stiffness = 3
        base_stiffness_main = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
        base_stiffness_fuselage = base_stiffness_main.copy()*sigma_fuselage
        base_stiffness_fuselage[4, 4] = base_stiffness_fuselage[5, 5]
        base_stiffness_tail = base_stiffness_main.copy()*sigma_tail
        base_stiffness_tail[4, 4] = base_stiffness_tail[5, 5]

        n_mass = 3
        base_mass_main = np.diag([m_bar_main, m_bar_main, m_bar_main, j_bar_main, 0.5*j_bar_main, 0.5*j_bar_main])
        base_mass_fuselage = np.diag([m_bar_fuselage,
                                      m_bar_fuselage,
                                      m_bar_fuselage,
                                      j_bar_fuselage,
                                      j_bar_fuselage*0.5,
                                      j_bar_fuselage*0.5])
        base_mass_tail = np.diag([m_bar_tail,
                                  m_bar_tail,
                                  m_bar_tail,
                                  j_bar_tail,
                                  j_bar_tail*0.5,
                                  j_bar_tail*0.5])

        # beam
        x = np.zeros((n_node, ))
        y = np.zeros((n_node, ))
        z = np.zeros((n_node, ))
        structural_twist = np.zeros((n_elem, n_node_elem))
        beam_number = np.zeros((n_elem, ), dtype=int)
        frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
        conn = np.zeros((n_elem, n_node_elem), dtype=int)
        stiffness = np.zeros((n_stiffness, 6, 6))
        elem_stiffness = np.zeros((n_elem, ), dtype=int)
        mass = np.zeros((n_mass, 6, 6))
        elem_mass = np.zeros((n_elem, ), dtype=int)
        boundary_conditions = np.zeros((n_node, ), dtype=int)
        app_forces = np.zeros((n_node, 6))

        stiffness[0, ...] = base_stiffness_main
        stiffness[1, ...] = base_stiffness_fuselage
        stiffness[2, ...] = base_stiffness_tail

        mass[0, ...] = base_mass_main
        mass[1, ...] = base_mass_fuselage
        mass[2, ...] = base_mass_tail

        we = 0
        wn = 0
        # inner right wing
        beam_number[we:we + n_elem_main1] = 0
        y[wn:wn + n_node_main1] = np.linspace(0.0, span_main1, n_node_main1)
        for ielem in range(n_elem_main1):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_main1] = 0
        elem_mass[we:we + n_elem_main1] = 0
        boundary_conditions[0] = 1
        app_forces[0] = [0, self.thrust, 0, 0, 0, 0]
        we += n_elem_main1
        wn += n_node_main1
        # outer right wing
        beam_number[we:we + n_elem_main1] = 0
        y[wn:wn + n_node_main2 - 1] = y[wn - 1] + np.linspace(0.0, np.cos(lambda_dihedral)*span_main2, n_node_main2)[1:]
        z[wn:wn + n_node_main2 - 1] = z[wn - 1] + np.linspace(0.0, np.sin(lambda_dihedral)*span_main2, n_node_main2)[1:]
        for ielem in range(n_elem_main2):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_main2] = 0
        elem_mass[we:we + n_elem_main2] = 0
        boundary_conditions[wn + n_node_main2 - 2] = -1
        we += n_elem_main2
        wn += n_node_main2 - 1
        # inner left wing
        beam_number[we:we + n_elem_main1 - 1] = 1
        y[wn:wn + n_node_main1 - 1] = np.linspace(0.0, -span_main1, n_node_main1)[1:]
        for ielem in range(n_elem_main1):
            conn[we + ielem, :] = ((np.ones((3, ))*(we+ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = 0
        elem_stiffness[we:we + n_elem_main1] = 0
        elem_mass[we:we + n_elem_main1] = 0
        we += n_elem_main1
        wn += n_node_main1 - 1
        # outer left wing
        beam_number[we:we + n_elem_main2] = 1
        y[wn:wn + n_node_main2 - 1] = y[wn - 1] + np.linspace(0.0, -np.cos(lambda_dihedral)*span_main2, n_node_main2)[1:]
        z[wn:wn + n_node_main2 - 1] = z[wn - 1] + np.linspace(0.0, np.sin(lambda_dihedral)*span_main2, n_node_main2)[1:]
        for ielem in range(n_elem_main2):
            conn[we + ielem, :] = ((np.ones((3, ))*(we+ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_main2] = 0
        elem_mass[we:we + n_elem_main2] = 0
        boundary_conditions[wn + n_node_main2 - 2] = -1
        we += n_elem_main2
        wn += n_node_main2 - 1
        # fuselage
        beam_number[we:we + n_elem_fuselage] = 2
        x[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, length_fuselage, n_node_fuselage)[1:]
        z[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, offset_fuselage, n_node_fuselage)[1:]
        for ielem in range(n_elem_fuselage):
            conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
        conn[we, 0] = 0
        elem_stiffness[we:we + n_elem_fuselage] = 1
        elem_mass[we:we + n_elem_fuselage] = 1
        we += n_elem_fuselage
        wn += n_node_fuselage - 1
        global end_of_fuselage_node
        end_of_fuselage_node = wn - 1
        # fin
        beam_number[we:we + n_elem_fin] = 3
        x[wn:wn + n_node_fin - 1] = x[end_of_fuselage_node]
        z[wn:wn + n_node_fin - 1] = z[end_of_fuselage_node] + np.linspace(0.0, fin_height, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_of_fuselage_node
        elem_stiffness[we:we + n_elem_fin] = 2
        elem_mass[we:we + n_elem_fin] = 2
        we += n_elem_fin
        wn += n_node_fin - 1
        end_of_fin_node = wn - 1
        # right tail
        beam_number[we:we + n_elem_tail] = 4
        x[wn:wn + n_node_tail - 1] = x[end_of_fin_node]
        y[wn:wn + n_node_tail - 1] = np.linspace(0.0, span_tail, n_node_tail)[1:]
        z[wn:wn + n_node_tail - 1] = z[end_of_fin_node]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_of_fin_node
        elem_stiffness[we:we + n_elem_tail] = 2
        elem_mass[we:we + n_elem_tail] = 2
        boundary_conditions[wn + n_node_tail - 2] = -1
        we += n_elem_tail
        wn += n_node_tail - 1
        # left tail
        beam_number[we:we + n_elem_tail] = 5
        x[wn:wn + n_node_tail - 1] = x[end_of_fin_node]
        y[wn:wn + n_node_tail - 1] = np.linspace(0.0, -span_tail, n_node_tail)[1:]
        z[wn:wn + n_node_tail - 1] = z[end_of_fin_node]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = end_of_fin_node
        elem_stiffness[we:we + n_elem_tail] = 2
        elem_mass[we:we + n_elem_tail] = 2
        boundary_conditions[wn + n_node_tail - 2] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        with h5.File(self.route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
            conectivities = h5file.create_dataset('connectivities', data=conn)
            num_nodes_elem_handle = h5file.create_dataset(
                'num_node_elem', data=n_node_elem)
            num_nodes_handle = h5file.create_dataset(
                'num_node', data=n_node)
            num_elem_handle = h5file.create_dataset(
                'num_elem', data=n_elem)
            stiffness_db_handle = h5file.create_dataset(
                'stiffness_db', data=stiffness)
            stiffness_handle = h5file.create_dataset(
                'elem_stiffness', data=elem_stiffness)
            mass_db_handle = h5file.create_dataset(
                'mass_db', data=mass)
            mass_handle = h5file.create_dataset(
                'elem_mass', data=elem_mass)
            frame_of_reference_delta_handle = h5file.create_dataset(
                'frame_of_reference_delta', data=frame_of_reference_delta)
            structural_twist_handle = h5file.create_dataset(
                'structural_twist', data=structural_twist)
            bocos_handle = h5file.create_dataset(
                'boundary_conditions', data=boundary_conditions)
            beam_handle = h5file.create_dataset(
                'beam_number', data=beam_number)
            app_forces_handle = h5file.create_dataset(
                'app_forces', data=app_forces)
            lumped_mass_nodes_handle = h5file.create_dataset(
                'lumped_mass_nodes', data=lumped_mass_nodes)
            lumped_mass_handle = h5file.create_dataset(
                'lumped_mass', data=lumped_mass)
            lumped_mass_inertia_handle = h5file.create_dataset(
                'lumped_mass_inertia', data=lumped_mass_inertia)
            lumped_mass_position_handle = h5file.create_dataset(
                'lumped_mass_position', data=lumped_mass_position)

        self.n_elem = n_elem
        self.n_node = n_node

        self.x = x
        self.y = y
        self.z = z

        self.n_elem_main = n_elem_main
        self.n_elem_fuselage = n_elem_fuselage
        self.n_elem_fin = n_elem_fin
        self.n_elem_tail = n_elem_tail

        self.n_node_main = n_node_main
        self.n_node_fuselage = n_node_fuselage
        self.n_node_fin = n_node_fin
        self.n_node_tail = n_node_tail

