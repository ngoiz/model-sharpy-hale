#! /usr/bin/env python3
import h5py as h5
import numpy as np
from structure import span_main
from sharpy.utils.geo_utils import generate_naca_camber


chord_main = 1.0
chord_tail = 0.5
chord_fin = 0.5
ea_main = 0.3
ea_fin = 0.5
ea_tail = 0.5

# reference area
area_ref = chord_main * 2 * span_main


class HaleAero:
    def __init__(self, m, structure, case_name, case_route, **kwargs):
        self.m = m
        self.structure = structure

        self.route = case_route
        self.case_name = case_name

        self.cs_deflection = kwargs.get('cs_deflection', 0.)
        self.rudder_deflection = kwargs.get('rudder_deflection', 0.)

        self.chord_main = chord_main
        self.chord_tail = chord_tail
        self.chord_fin = chord_fin

    def generate(self):

        n_surfaces = 5
        structure = self.structure

        n_elem = structure.n_elem
        n_node_elem = structure.n_node_elem
        n_control_surfaces = 2
        n_elem_main = structure.n_elem_main
        n_node_main = structure.n_node_main
        m = self.m
        n_elem_fuselage = structure.n_elem_fuselage
        n_node_fuselage = structure.n_node_fuselage
        n_elem_fin = structure.n_elem_fin
        n_node_fin = structure.n_node_fin
        n_elem_tail = structure.n_elem_tail
        n_node_tail = structure.n_node_tail

        # aero
        airfoil_distribution = np.zeros((structure.n_elem, structure.n_node_elem), dtype=int)
        surface_distribution = np.zeros((structure.n_elem,), dtype=int) - 1
        surface_m = np.zeros((n_surfaces, ), dtype=int)
        m_distribution = 'uniform'
        aero_node = np.zeros((structure.n_node,), dtype=bool)
        twist = np.zeros((structure.n_elem, structure.n_node_elem))
        sweep = np.zeros((structure.n_elem, structure.n_node_elem))
        chord = np.zeros((structure.n_elem, structure.n_node_elem,))
        elastic_axis = np.zeros((structure.n_elem, structure.n_node_elem,))

        control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1
        control_surface_type = np.zeros((n_control_surfaces, ), dtype=int)
        control_surface_deflection = np.zeros((n_control_surfaces, ))
        control_surface_chord = np.zeros((n_control_surfaces, ), dtype=int)
        control_surface_hinge_coord = np.zeros((n_control_surfaces, ), dtype=float)

        # control surface type 0 = static
        # control surface type 1 = dynamic
        control_surface_type[0] = 0
        control_surface_deflection[0] = self.cs_deflection
        control_surface_chord[0] = m
        control_surface_hinge_coord[0] = -0.25 # nondimensional wrt elastic axis (+ towards the trailing edge)

        control_surface_type[1] = 0
        control_surface_deflection[1] = self.rudder_deflection
        control_surface_chord[1] = m // 2
        control_surface_hinge_coord[1] = -0. # nondimensional wrt elastic axis (+ towards the trailing edge)

        we = 0
        wn = 0
        # right wing (surface 0, beam 0)
        i_surf = 0
        airfoil_distribution[we:we + n_elem_main, :] = 0
        surface_distribution[we:we + n_elem_main] = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_main] = True
        temp_chord = np.linspace(chord_main, chord_main, n_node_main)
        temp_sweep = np.linspace(0.0, 0*np.pi/180, n_node_main)
        node_counter = 0
        for i_elem in range(we, we + n_elem_main):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = temp_chord[node_counter]
                elastic_axis[i_elem, i_local_node] = ea_main
                sweep[i_elem, i_local_node] = temp_sweep[node_counter]

        we += n_elem_main
        wn += n_node_main

        # left wing (surface 1, beam 1)
        i_surf = 1
        airfoil_distribution[we:we + n_elem_main, :] = 0
        # airfoil_distribution[wn:wn + n_node_main - 1] = 0
        surface_distribution[we:we + n_elem_main] = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_main - 1] = True
        # chord[wn:wn + num_node_main - 1] = np.linspace(main_chord, main_tip_chord, num_node_main)[1:]
        # chord[wn:wn + num_node_main - 1] = main_chord
        # elastic_axis[wn:wn + num_node_main - 1] = main_ea
        temp_chord = np.linspace(chord_main, chord_main, n_node_main)
        node_counter = 0
        for i_elem in range(we, we + n_elem_main):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = temp_chord[node_counter]
                elastic_axis[i_elem, i_local_node] = ea_main
                sweep[i_elem, i_local_node] = -temp_sweep[node_counter]

        we += n_elem_main
        wn += n_node_main - 1

        we += n_elem_fuselage
        wn += n_node_fuselage - 1 - 1
        #
        # # fin (surface 2, beam 3)
        i_surf = 2
        airfoil_distribution[we:we + n_elem_fin, :] = 1
        # airfoil_distribution[wn:wn + n_node_fin] = 0
        surface_distribution[we:we + n_elem_fin] = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_fin] = True
        # chord[wn:wn + num_node_fin] = fin_chord
        for i_elem in range(we, we + n_elem_fin):
            for i_local_node in range(n_node_elem):
                chord[i_elem, i_local_node] = chord_fin
                elastic_axis[i_elem, i_local_node] = ea_fin
                control_surface[i_elem, i_local_node] = 1
        # twist[end_of_fuselage_node] = 0
        # twist[wn:] = 0
        # elastic_axis[wn:wn + num_node_main] = fin_ea
        we += n_elem_fin
        wn += n_node_fin - 1
        control_surface[we - 1, :] = -1
        #
        # # # right tail (surface 3, beam 4)
        i_surf = 3
        airfoil_distribution[we:we + n_elem_tail, :] = 2
        # airfoil_distribution[wn:wn + n_node_tail] = 0
        surface_distribution[we:we + n_elem_tail] = i_surf
        surface_m[i_surf] = m
        # XXX not very elegant
        aero_node[wn:] = True
        # chord[wn:wn + num_node_tail] = tail_chord
        # elastic_axis[wn:wn + num_node_main] = tail_ea
        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                twist[i_elem, i_local_node] = -0
        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                chord[i_elem, i_local_node] = chord_tail
                elastic_axis[i_elem, i_local_node] = ea_tail
                control_surface[i_elem, i_local_node] = 0

        we += n_elem_tail
        wn += n_node_tail
        #
        # # left tail (surface 4, beam 5)
        i_surf = 4
        airfoil_distribution[we:we + n_elem_tail, :] = 2
        # airfoil_distribution[wn:wn + n_node_tail - 1] = 0
        surface_distribution[we:we + n_elem_tail] = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_tail - 1] = True
        # chord[wn:wn + num_node_tail] = tail_chord
        # elastic_axis[wn:wn + num_node_main] = tail_ea
        # twist[we:we + num_elem_tail] = -tail_twist
        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                twist[i_elem, i_local_node] = -0
        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                chord[i_elem, i_local_node] = chord_tail
                elastic_axis[i_elem, i_local_node] = ea_tail
                control_surface[i_elem, i_local_node] = 0
        we += n_elem_tail
        wn += n_node_tail

        with h5.File(self.route + '/' + self.case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
                generate_naca_camber(P=0, M=0)))
            naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
                generate_naca_camber(P=0, M=0)))
            naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
                generate_naca_camber(P=0, M=0)))

            # chord
            chord_input = h5file.create_dataset('chord', data=chord)
            dim_attr = chord_input .attrs['units'] = 'm'

            # twist
            twist_input = h5file.create_dataset('twist', data=twist)
            dim_attr = twist_input.attrs['units'] = 'rad'

            # sweep
            sweep_input = h5file.create_dataset('sweep', data=sweep)
            dim_attr = sweep_input.attrs['units'] = 'rad'

            # airfoil distribution
            airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

            surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
            surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
            m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

            aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
            elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

            control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
            control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
            control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
            control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord', data=control_surface_hinge_coord)
            control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)

