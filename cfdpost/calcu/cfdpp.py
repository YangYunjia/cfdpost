import os
import math
import numpy as np
from .util import *
from .system import cfdpp_cmd
from py2tec.py2tec import tec2py

class cfdpp():
    ''' 
    operation interface to CFD++
    '''

    def __init__(self, op_dir=None, core=1):
        if op_dir is None:
            op_dir = os.getcwd()

        self.set_path(op_dir)

        self.core_number = core

        self.FFM_data = None
        self.areas = None

    def set_path(self, new_path):
        self.op_dir = new_path
        self.inp_dir = new_path + "//mcfd.inp"
        self.bak_dir = new_path + "//mcfd_bak_sp.inp"
        self.FFM_data = None
        self.areas = None

        if not os.path.exists(self.inp_dir):
            print("mcfd.inp not exist in " + self.op_dir + "nbc not set")
        else:
            self.bc_number = int(self.read_para('mbcons'))
        
        print("\ndirection changed to " + self.op_dir)

        os.chdir(self.op_dir)

    def metis(self):
        os.system("cd %s && @tometis pmetis %d" % (self.op_dir, self.core_number))

    
    # @staticmethod
    def set_para(self, key, value):

        data = ''

        try:
            val_str = str(value)
        except:
            print("value cant be convert to a string")

        with open(self.inp_dir, 'r') as f, open(self.bak_dir, 'w') as fbak:
            for line in f.readlines():
                fbak.write(line)

                # Slit boundary type
                
                if line.find(key) > -1:
                    line = key + " " +val_str + "\n"
                
                data += line
        
        with open(self.inp_dir, 'w') as f:
            f.writelines(data)

    def read_para(self, key):
        
        with open(self.inp_dir, 'r') as f:
            for line in f.readlines(): 
                if line.find(key) > -1:
                    return line.split()[1]
                

    # @staticmethod
    def set_infset(self, inf_num, values, filte=[]):
        
        data = ''

        # try:
        #     val_str = str(value)
        # except:
        #     print("value cant be convert to a string")

        with open(self.inp_dir, 'r') as f, open(self.bak_dir, 'w') as fbak:
            flag = False
            var_idx = 0
            for line in f.readlines():
                fbak.write(line)

                # Slit boundary type
                
                if line.find('seq.# %s' % str(inf_num)) > -1:
                    splitline = line.split()
                    value_num = int(splitline[3])
                    rest_value_num = value_num
                    inf_name = splitline[5]

                    if len(values) != value_num:
                        print("number not match")
                    else:
                        flag = True
            
                elif flag:
                    pre_data = line.split()
                    line = 'values '
                    for i in range(min(5, rest_value_num)):
                        if var_idx in filte:
                            line += pre_data[i + 1] + " "
                        else:
                            line += "%.4e " % values[var_idx]
                        var_idx += 1
                    line += "\n"
                    rest_value_num -= 5
                    if rest_value_num <= 0:
                        flag = False
                
                data += line
        
        with open(self.inp_dir, 'w') as f:
            f.writelines(data)

    def run_cfd(self, restart=False, step=1500):

        self.set_para("istart", int(restart))
        self.set_para("ntstep", step)

        print("runing cfd with core number %d" % self.core_number)

        if self.core_number > 1:
            os.system('start /wait /min "" "C:\Program Files\MPICH2\\bin\mpiexec.exe" -localonly -np %d mpimcfd' % self.core_number)

    def read_FFM_history(self, n_var=8, n_step=1e10):

        with open(self.op_dir + "//mcfd.info1", 'r') as f:
            lines = f.readlines()
        
        idx = 25
        step = 0
        file_len = len(lines)

        n_bc = self.bc_number
        n_step = min(int((file_len - 11) / (23 * n_bc + 1)), n_step)
        print("Acquiring %d bcs intergal data for first %d steps" % (n_bc, n_step))

        data = np.zeros((n_step, n_bc, n_var))
        areass = np.zeros((n_bc, 4))

        for _ in range(n_step):
            for i_bc in range(n_bc):
                for i_var in range(n_var):
                    data[step, i_bc, i_var] = lines[idx].split()[2]
                    idx += 1
                idx += 15
            idx += 1
            step += 1
        
        for i_area in range(n_bc):
            areas_str = lines[33 + i_area * 23].split()
            for i_typ in range(4):
                areass[i_area, i_typ] = float(areas_str[1 + i_typ])


        self.FFM_data = data
        self.areas = areass
        # return data, areass
        

    def read_flux(self, typ, bc_series):
        if self.FFM_data is None:
            self.read_FFM_history()
        
        if typ == 'energy':
            int_typ = 0
        elif typ == 'mass':
            int_typ = 1
        elif typ == 'fx':
            int_typ = 2
        elif typ == 'fy':
            int_typ = 3
        elif typ == 'fz':
            int_typ = 4
        
        eps = 1e-3
        flux = 0.0
        for i_bc in bc_series:
            # print("reading bc No. %d, type %s" % (i_bc,typ))

            flux_i = self.FFM_data[-1, i_bc-1, int_typ] 
            if abs(flux_i - self.FFM_data[-5, i_bc-1, int_typ]) / flux_i > eps:
                print("bc No. %d, type %s not converge" % (i_bc,typ))
            flux += flux_i
        
        return flux

    def read_area(self, typ, bc_series):
        if self.areas is None:
            self.read_FFM_history()

        if typ == 'x':
            int_typ = 0
        elif typ == 'y':
            int_typ = 1
        elif typ == 'z':
            int_typ = 2
        
        area = 0.0
        for i_bc in bc_series:
            area += self.areas[i_bc-1, int_typ]

        return area

    def extract_bc(self, bc_series):
        data = {'varnames': None, 'lines': []}
        for i in bc_series:
            cfdpp_cmd("exbc2do1 exbcsin.bin pltosout.bin %d" % i)
            if not os.path.exists("BC%d.mpf1d" % i):
                print("    [Warning] BC%d not extract" %i)
                return
            data_tmp = tec2py(os.path.join(self.op_dir, "BC%d.dat" % i))
            if data['varnames'] is None:
                data['varnames'] = data_tmp['varnames']
            data['lines'] += data_tmp['lines']
        
        return data

            