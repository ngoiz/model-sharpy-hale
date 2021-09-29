#! /usr/bin/env python3
import h5py as h5
import configobj
import numpy as np
from structure import HaleStructure
from aero import HaleAero
import os
import sharpy.sharpy_main


class Hale:

    def __init__(self, case_name, case_route, output_route):
        self.case_name = case_name
        self.case_route = case_route
        self.output_route = output_route

        self.structure = None
        self.aero = None

        self.settings = None

    def init_structure(self, **kwargs):
        self.structure = HaleStructure(self.case_name, self.case_route, **kwargs)

    def init_aero(self, m, **kwargs):
        self.aero = HaleAero(m, self.structure, self.case_name, self.case_route, **kwargs)

    def set_flight_controls(self, thrust=0., elevator=0., rudder=0.):
        self.structure.set_thrust(thrust)

        if self.aero is not None:
            self.aero.cs_deflection = elevator
            self.aero.rudder_deflection = rudder

    def generate(self):

        if not os.path.isdir(self.case_route):
            os.makedirs(self.case_route)

        self.structure.generate()

        if self.aero is not None:
            self.aero.generate()

    def create_settings(self, settings):
        file_name = self.case_route + '/' + self.case_name + '.sharpy'
        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()
        self.settings = settings

    def clean(self):
        fem_file_name = self.case_route + '/' + self.case_name + '.fem.h5'
        if os.path.isfile(fem_file_name):
            os.remove(fem_file_name)

        dyn_file_name = self.case_route + '/' + self.case_name + '.dyn.h5'
        if os.path.isfile(dyn_file_name):
            os.remove(dyn_file_name)

        aero_file_name = self.case_route + '/' + self.case_name + '.aero.h5'
        if os.path.isfile(aero_file_name):
            os.remove(aero_file_name)

        solver_file_name = self.case_route + '/' + self.case_name + '.sharpy'
        if os.path.isfile(solver_file_name):
            os.remove(solver_file_name)

        flightcon_file_name = self.case_route + '/' + self.case_name + '.flightcon.txt'
        if os.path.isfile(flightcon_file_name):
            os.remove(flightcon_file_name)

    def run(self):
        sharpy.sharpy_main.main(['', self.case_route + '/' + self.case_name + '.sharpy'])
