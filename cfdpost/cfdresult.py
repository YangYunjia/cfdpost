'''
Post process of CFD results
'''
import copy
axis_2_idx = {'x': 3, 'y': 4, 'z': 5}

import os
import platform

import numpy as np
import struct as st

from typing import Tuple, List

class cfl3d():
    '''
    Extracting data from cfl3d results
    '''

    def __init__(self):
        print('All static method functions')
        pass

    @staticmethod
    def readCoef(path: str, n=100, output_error=False):
        '''
        Read clcd_wall.dat or clcd.dat of the CFL3D outputs.

        >>> converge, CL, CD, Cm, CDp, CDf = readCoef(path: str, n=100, output_error=False)
        >>> converge, CL, CD, Cm, CDp, CDf, errs = readCoef(path: str, n=100, output_error=True)

        ### Inputs:
        ```text
        path:   folder that contains the results
        n:      get the mean value of final n steps
        ```

        ### Return:
        ```text
        converge (bool), CL, CD, Cm(z), CDp, CDf
        errs = [err_CL, err_CD, err_Cm, err_CDp, err_CDf]
        ```
        '''
        converge = True
        CL = 0.0
        CD = 0.0
        Cm = 0.0
        CDp = 0.0
        CDf = 0.0
        errs = [0.0 for _ in range(5)]

        out1 = os.path.join(path, 'clcd.dat')
        out2 = os.path.join(path, 'clcd_wall.dat')

        if os.path.exists(out1):
            out = out1
        elif os.path.exists(out2): 
            out = out2
        else:
            if output_error:
                return False, CL, CD, Cm, CDp, CDf, errs
            else:
                return False, CL, CD, Cm, CDp, CDf

        CLs = np.zeros(n)
        CDs = np.zeros(n)
        Cms = np.zeros(n)
        CDps = np.zeros(n)
        CDfs = np.zeros(n)
        with open(out, 'r') as f:
            lines = f.readlines()
            n_all = len(lines)

            i = 1
            k = 0
            while i<n_all-4 and k<n:

                L1 = lines[-i].split()
                L2 = lines[-i-1].split()
                i += 1

                if L1[2] == L2[2]:
                    # Duplicated lines of the final step when using multiple blocks
                    continue

                CLs[k] = float(L1[5])
                CDs[k] = float(L1[6])
                Cms[k] = float(L1[12])
                CDps[k] = float(L1[8])
                CDfs[k] = float(L1[9])
                k += 1

        CL_  = np.mean(CLs)
        if k < n*0.5:
            converge = False

        elif np.max(CLs)-np.min(CLs) < max(0.01, 0.01*CL_):
            CL = CL_
            CD = np.mean(CDs)
            Cm = np.mean(Cms)
            CDp = np.mean(CDps)
            CDf = np.mean(CDfs)
            errs[0] = np.max(CLs)-np.min(CLs)
            errs[1] = np.max(CDs)-np.min(CDs)
            errs[2] = np.max(Cms)-np.min(Cms)
            errs[3] = np.max(CDps)-np.min(CDps)
            errs[4] = np.max(CDfs)-np.min(CDfs)

        else:
            converge = False
        
        if output_error:
            return converge, CL, CD, Cm, CDp, CDf, errs
        else:
            return converge, CL, CD, Cm, CDp, CDf

    @staticmethod
    def readCoef1(path: str, n: int = 100, output_error: bool = False, conv_hold: float = 0.01):
        '''
        Read clcd_wall.dat or clcd.dat of the CFL3D outputs.

        >>> converge, CL, CD, Cm, CDp, CDf = readCoef(path: str, n=100, output_error=False)
        >>> converge, CL, CD, Cm, CDp, CDf, errs = readCoef(path: str, n=100, output_error=True)

        ### Inputs:
        ```text
        path:   folder that contains the results
        n:      get the mean value of final n steps
        ```

        ### Return:
        ```text
        converge (bool), steps,
        vals = [CL, CD, Cm(z), CDp, CDf]
        errs = [err_CL, err_CD, err_Cm, err_CDp, err_CDf]
        ```
        '''
        converge = 0
        steps = 0
        vals = np.zeros(5)
        errs = np.zeros(5)

        out1 = os.path.join(path, 'clcd.dat')
        out2 = os.path.join(path, 'clcd_wall.dat')

        if os.path.exists(out1):
            out = out1
        elif os.path.exists(out2): 
            out = out2
        else:
            if output_error:
                return -1, 0, vals, errs
            else:
                return -1, 0, vals

        CLs = np.zeros((5, n))

        with open(out, 'r') as f:
            lines = f.readlines()
            n_all = len(lines)
            if n_all < 4:
                if output_error:
                    return -1, 0, vals, errs
                else:
                    return -1, 0, vals

        steps = int(lines[-1].split()[2])
        i = 1
        k = 0
        while i<n_all-4 and k<n:

            L1 = lines[-i].split()
            i += 1

            if int(L1[2]) == steps:
                # Duplicated lines of the final step when using multiple blocks
                continue

            CLs[0, k] = float(L1[5])
            CLs[1, k] = float(L1[6])
            CLs[2, k] = float(L1[12])
            CLs[3, k] = float(L1[8])
            CLs[4, k] = float(L1[9])
            k += 1

        vals = np.mean(CLs, axis=1)

        if k < n*0.5:
            converge = -2
        else:
            errs = np.max(CLs, axis=1)-np.min(CLs, axis=1)

            if errs[0] > max(conv_hold, conv_hold * vals[0]):
                converge = 1
        
        if output_error:
            return converge, steps, vals, errs
        else:
            return converge, steps, vals

    @staticmethod
    def readAoA(path: str, n=100, output_error=False):
        '''
        Read cfl3d.alpha of the CFL3D outputs.

        >>> succeed, AoA = readAoA(path: str, n=100, output_error=False)
        >>> succeed, AoA, err = readAoA(path: str, n=100, output_error=True)

        ### Inputs:
        ```text
        path:   folder that contains the results
        n:      get the mean value of final n steps
        ```

        ### Return:
        ```text
        succeed (bool), AoA
        ```
        '''
        succeed = True
        AoA = 0.0

        if platform.system() in 'Windows':
            out = path+'\\cfl3d.alpha'
        else:
            out = path+'/cfl3d.alpha'

        if not os.path.exists(out):
            if output_error:
                return False, AoA, 0.0
            else:
                return False, AoA

        AoAs = np.zeros(n)
        with open(out, 'r') as f:
            lines = f.readlines()

            if len(lines)<=n+10:
                f.close()
                if output_error:
                    return False, AoA, 0.0
                else:
                    return False, AoA

            for k in range(n):
                L1 = lines[-k-1].split()
                AoAs[k] = float(L1[3])

        AoA = np.mean(AoAs)

        if output_error:
            return succeed, AoA, np.max(AoAs)-np.min(AoAs)
        else:
            return succeed, AoA

    @staticmethod
    def readinput(path: str):
        '''
        Read cfl3d.inp of the CFL3D input.

        >>> succeed, Minf, AoA0, Re, l2D = readinput(path: str)

        ### Inputs:
        ```text
        path:   folder that contains the input files
        ```

        ### Return:
        ```text
        succeed (bool), Minf, AoA0 (deg), Re (e6, /m), l2D(bool)
        ```
        '''

        succeed = True
        Minf = 0.0
        AoA0 = 0.0
        Re = 0.0
        l2D = False

        if platform.system() in 'Windows':
            inp = path+'\\cfl3d.inp'
        else:
            inp = path+'/cfl3d.inp'

        if not os.path.exists(inp):
            print(inp)
            return False, Minf, AoA0, Re, l2D

        with open(inp, 'r') as f:
            lines = f.readlines()

            for i in range(len(lines)-1):
                line = lines[i].split()

                if 'XMACH' in line[0]:
                    L1 = lines[i+1].split()
                    Minf = float(L1[0])
                    AoA0 = float(L1[1])
                    Re   = float(L1[3])

                if 'NGRID' in line[0]:
                    L1 = lines[i+1].split()
                    if int(L1[5])==1:
                        l2D = True

        return succeed, Minf, AoA0, Re, l2D

    @staticmethod
    def readprt(path: str, fname: str = 'cfl3d.prt', write_to_file: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        '''
        Read cfl3d.prt of the CFL3D output. (python version of the original Fortran file)

        >>> succeed = readprt(path: str, fname='cfl3d.prt')

        ### Inputs:
        ```text
        path:   folder that contains the output files
        ```

        ### Return:
        ```text
        (block_p, block_v)
            each contains several blocks, each block is a np.ndarray of size (Nx * Ny * Nz * Nv)
        ```
        '''
        mi = 10000000   # maximum size of i*j*k

        def read_block(fin):

            xyz = []
            # read pressure fields
            ijk_i = np.zeros((3,), dtype=int)
            ijkmax = np.zeros((3,), dtype=int)
            ijkmin = np.ones((3,), dtype=int) * mi
            while True:
                new_line_split = fin.readline().split()
                if len(new_line_split) == 0 or new_line_split[0] in ['I']:
                    # end of the first part
                    break
                # read the line 
                for i in range(3):
                    ijk_i[i] = int(new_line_split[i])
                
                xyz_i = []
                for i in range(11):
                    #  X, Y, Z, U/Uinf, V/Vinf, W/Winf, P/Pinf, T/Tinf, MACH, cp, tur. vis. （11）
                    #           ^       ^       ^                       ^  this four should be zero!
                    # the value at cell center!
                    xyz_i.append(float(new_line_split[i + 3]))
                xyz.append(xyz_i)
                
                # update the block size (the size is not written in the file)
                ijkmax = np.maximum(ijkmax, ijk_i)
                ijkmin = np.minimum(ijkmin, ijk_i)
            
            block_shape = (ijkmax - ijkmin + 1).tolist()
            xyz_p = np.array(xyz).reshape(block_shape + [-1])

            # read viscous fields (the data may not be full)
            xyz_v = np.zeros(block_shape + [12])
            while True:
                aline = fin.readline()
                if aline == '': break   # End of file

                new_line_split = aline.split()
                if len(new_line_split) == 0: continue   # null line
                if new_line_split[0] in ['BLOCK']: break # go into next block

                if new_line_split[0] in ['I'] and new_line_split[6] in 'dn':
                    while True:
                        new_line_split = fin.readline().split()
                        if len(new_line_split) == 0:
                            break
                        for i in range(3):
                            ijk_i[i] = int(new_line_split[i])
                        ijk_i = ijk_i - ijkmin
                        for i in range(12):
                            #  X, Y, Z, dn, P/Pinf, T/Tinf, Cf, Ch, yplus, Cfx, Cfy, Cfz （12）
                            # the value at cell center!
                            xyz_v[tuple(ijk_i)][i] = float(new_line_split[i + 3])

            return xyz_p, xyz_v

        def write_block(block, fout, num_v, block_name):
            bs = block.shape
            print(bs)
            if bs[0] == 1: fout.write('zone T="%s" i= %d j= %d k= %d \n'%(block_name, bs[0], bs[1], bs[2]))
            if bs[1] == 1: fout.write('zone T="%s" i= %d j= %d k= %d \n'%(block_name, bs[2], bs[1], bs[0]))
            if bs[2] == 1: fout.write('zone T="%s" i= %d j= %d k= %d \n'%(block_name, bs[1], bs[0], bs[2]))
            for _i, _j, _k in np.ndindex(bs[:3]):
                out_line = '%18.10f %18.10f %18.10f' % (block[_i, _j, _k, 0], block[_i, _j, _k, 1], block[_i, _j, _k, 2])
                out_line += ' %5d %5d %5d'%(_i, _j, _k)
                for _v in range(num_v):
                    out_line += ' %18.10f'%(block[_i, _j, _k, 3 + _v])
                
                fout.write(out_line + '\n')

        if platform.system() in 'Windows':
            prt  = path+'\\'+fname
            out1 = path+'\\surface.dat'
            out2 = path+'\\surface2.dat'
        else:
            prt  = path+'/'+fname
            out1 = path+'/surface.dat'
            out2 = path+'/surface2.dat'

        if not os.path.exists(prt):
            raise IOError('File %s not exist!' % prt)
        
        block_p = []        # pressure distributions for blocks
        block_v = []        # viscous distributions for blocs

        with open(prt, 'r') as f0:

            while True:

                line = f0.readline()
                if line == '':  break

                line = line.split()
                if len(line) == 0: continue
                if not line[0] in 'I': continue
                
                if line[6] in 'U/Uinf':
                    #* Pressure distribution (first part of the .prt file)
                    block = read_block(fin=f0)
                    block_p.append(block[0])
                    block_v.append(block[1])
        
        if write_to_file:
            with open(out1, 'w') as f1, open(out2, 'w') as f2:
                f1.write('Variables = X Y Z I J K U V W P T M Cp ut \n')
                f2.write('Variables = X Y Z I J K dn P T Cf Ch yplus Cfx Cfy Cfz\n')
                for idx in range(len(block_p)):
                    write_block(block=block_p[idx], fout=f1, num_v=8, block_name=str(idx))
                    write_block(block=block_v[idx], fout=f2, num_v=9, block_name=str(idx))

        return block_p, block_v

    @staticmethod
    def readprt_foil(path: str, j0: int, j1: int, fname='cfl3d.prt', coordinate='xy'):
        '''
        Read and extract foil Cp from cfl3d.prt

        >>> succeed, field, foil = readprt_foil(path: str, j0: int, j1: int, fname='cfl3d.prt')

        ### Inputs:
        ```text
        path:   folder that contains the output files
        j0:     j index of the lower surface TE
        j1:     j index of the upper surface TE + 1
        ```

        ## cfl3d.prt index
        ```text
        i : 1 - 1   symmetry plane
        j : 1 - nj  from far field of lower surface TE to far field of upper surface TE
        k : 1 - nk  from surface to far field
        ```

        ### Return:
        ```text
        succeed (bool), (field: X,Y,U,V,P,T,Ma,Cp,vi), (foil: x, y, P, T, Cf), (freestream: mach, alph, beta, reue, tinf, time)
        ```
        
        ### Raise:
        ```text
        FileNotFoundError
        ```
        '''

        if platform.system() in 'Windows':
            prt  = path+'\\'+fname
        else:
            prt  = path+'/'+fname

        X = None
        axis_2_idx = {'x': 3, 'y': 4, 'z': 5}
        x_idx = axis_2_idx[coordinate[0]]
        y_idx = axis_2_idx[coordinate[1]]

        f0 = open(prt, 'r')

        # read freestream data
        for _ in range(4): line = f0.readline()
        freestream = [float(i) for i in line.split()]

        counter = 0
        while True:

            line = f0.readline()
            if line == '':
                break

            line = line.split()
            if len(line) == 0:
                continue

            if 'BLOCK' in line[0]:
                ni = int(line[-3])
                nj = int(line[-2])
                nk = int(line[-1])
                continue

            if line[0] in 'I':

                if counter == 0: # field
                    counter += 1
                    X = np.zeros([nj, nk])
                    Y = np.zeros([nj, nk])
                    U = np.zeros([nj, nk])
                    V = np.zeros([nj, nk])
                    P = np.zeros([nj, nk])
                    T = np.zeros([nj, nk])
                    Ma = np.zeros([nj, nk])
                    Cp = np.zeros([nj, nk])
                    vi = np.zeros([nj, nk])

                    x_idx = axis_2_idx[coordinate[0]]
                    y_idx = axis_2_idx[coordinate[1]]

                    for k in range(nk):
                        for j in range(nj):
                            L1 = f0.readline()
                            L1 = L1.split()

                            X [j,k] = float(L1[x_idx])
                            Y [j,k] = float(L1[y_idx])
                            U [j,k] = float(L1[6])
                            V [j,k] = float(L1[7])
                            P [j,k] = float(L1[9])
                            T [j,k] = float(L1[10])
                            Ma[j,k] = float(L1[11])
                            Cp[j,k] = float(L1[12])
                            vi[j,k] = float(L1[13])

                elif counter == 1:
                    counter += 1
                    Xsurf = np.zeros((nj,))
                    Ysurf = np.zeros((nj,))
                    Psurf = np.zeros((nj,))
                    Tsurf = np.zeros((nj,))
                    Cfsurf = np.zeros((nj,))

                    for j in range(2, nj):
                        L1 = f0.readline().split()
                        Xsurf[j-1] = float(L1[x_idx])
                        Ysurf[j-1] = float(L1[y_idx])
                        Psurf[j-1] = float(L1[7])
                        Tsurf[j-1] = float(L1[8])
                        Cfsurf[j-1] = float(L1[9])
                    
                    break

        if X is None:
            return False, None, None

        field = (X,Y,U,V,P,T,Ma,Cp,vi)
        f0.close()

        Psurf = (Psurf - 1.) / (0.5 * freestream[0]**2 * 1.4)

        foil = (Xsurf[j0:j1], Ysurf[j0:j1], Psurf[j0:j1], Tsurf[j0:j1], Cfsurf[j0:j1])

        return True, field, foil, freestream

    @staticmethod
    def foildata(field_data: np.array, j0: int, j1: int):
        '''
        Extract wall data from field data

        >>> data = field_data[j0:j1,0]

        ### Inputs:
        ```text
        field_data: ndarray [nj,nk]
        j0:         j index of the lower surface TE
        j1:         j index of the upper surface TE + 1
        ```

        ## cfl3d.prt index
        ```text
        i : 1 - 1   symmetry plane
        j : 1 - nj  from far field of lower surface TE to far field of upper surface TE
        k : 1 - nk  from surface to far field
        ```
        '''
        return field_data[j0:j1,0]

    @staticmethod
    def readPlot2d(path: str, fname_grid='plot3d_grid.xyz', fname_sol='plot3d_sol.bin', binary=True, _double_precision=True):
        '''
        Plot3D Format grid and solution:
        2D, Whole, Formatted, Single-Block Grid and Solution

        https://www.grc.nasa.gov/www/wind/valid/plot3d.html

        >>> xy, qq, mach, alfa, reyn = readPlot2d(path: str, 
        >>>         fname_grid='plot3d_grid.xyz', fname_sol='plot3d_sol.bin', binary=True)

        ### Input:
        ```text
        path:       folder that contains the output files
        fname_grid: grid file name
        fname_sol:  solution file name
        binary:     binary or ASCII format
        ```

        ### Return:
        ```text
        xy:     ndarray [ni,nj,2], or None
        qq:     ndarray [ni,nj,4], or None
                non-dimensionalized RHO, RHO-U, RHO-V, E
                q1: density by the reference density, rho
                q*: velocity by the reference speed of sound, ar
                q4: total energy per unit volume by rho*ar^2
        mach:   freestream Mach number, = ur/ar
                ur: reference velocity
        alfa:   freestream angle-of-attack
        reyn:   freestream Reynolds number, = rho*ar*Lr/miur
                Lr: reference length
                miur: reference viscosity
        ```
        '''
        xy = None
        qq = None
        mach = 0.0
        alfa = 0.0
        reyn = 0.0

        if _double_precision:
            r_format = 8
            s_format = 'd'
        else:
            r_format = 4
            s_format = 'f'

        if binary:

            with open(os.path.join(path, fname_grid), 'rb') as f:

                a,  = st.unpack('i', f.read(4))
                ni, = st.unpack('i', f.read(4))
                nj, = st.unpack('i', f.read(4))
                xy  = np.zeros((ni,nj,2))

                for v in range(2):
                    for j in range(nj):
                        for i in range(ni):
                            xy[i,j,v], = st.unpack(s_format, f.read(r_format))

            with open(os.path.join(path, fname_sol), 'rb') as f:

                _,  = st.unpack('i', f.read(4))
                ni, = st.unpack('i', f.read(4))
                nj, = st.unpack('i', f.read(4))
                qq  = np.zeros((ni,nj,4))

                mach, = st.unpack(s_format, f.read(r_format))   # freestream Mach number
                alfa, = st.unpack(s_format, f.read(r_format))   # freestream angle-of-attack
                reyn, = st.unpack(s_format, f.read(r_format))   # freestream Reynolds number
                time, = st.unpack(s_format, f.read(r_format))   # time

                for q in range(4):
                    for j in range(nj):
                        for i in range(ni):
                            qq[i,j,q], = st.unpack(s_format, f.read(r_format))


        else:

            with open(os.path.join(path, fname_grid), 'r') as f:
                lines = f.readlines()

                line = lines[1].split()
                ni = int(line[0])
                nj = int(line[1])
                xy = np.zeros((ni,nj,2))

                k_line = 2
                k_item = 0
                line   = lines[k_line].split()
                len_line = len(line)
                data = [float(a) for a in line]

                for k in range(2):
                    for j in range(nj):
                        for i in range(ni):
                            # Read next line
                            if k_item >= len_line:
                                k_line += 1
                                k_item = 0
                                line = lines[k_line].split()
                                len_line = len(line)
                                data = [float(a) for a in line]

                            # Assign to xx, yy
                            xy[i,j,k] = data[k_item]
                            k_item += 1

            with open(os.path.join(path, fname_sol), 'r') as f:
                lines = f.readlines()

                line = lines[1].split()
                ni = int(line[0])
                nj = int(line[1])
                qq = np.zeros((ni,nj,4))

                line = lines[2].split()
                mach = float(line[0])   # freestream Mach number
                alfa = float(line[1])   # freestream angle-of-attack
                reyn = float(line[2])   # freestream Reynolds number
                time = float(line[3])   # time

                k_line = 3
                k_item = 0
                line   = lines[k_line].split()
                len_line = len(line)
                data = [float(a) for a in line]

                for n in range(4):
                    for j in range(nj):
                        for i in range(ni):
                            # Read next line
                            if k_item >= len_line:
                                k_line += 1
                                k_item = 0
                                line = lines[k_line].split()
                                len_line = len(line)
                                data = [float(a) for a in line]

                            # Assign to qq
                            qq[i,j,n] = data[k_item]
                            k_item += 1


        return xy, qq, mach, alfa, reyn

    @staticmethod
    def readPlot3d(path: str, fname_grid='plot3d_grid.xyz', fname_sol='plot3d_sol.bin', binary=True, _double_precision=True):
        '''
        Plot3D Format grid and solution:
        3D, Whole, Unformatted, Multi-Block Grid and Solution

        https://www.grc.nasa.gov/www/wind/valid/plot3d.html

        >>> xyz, qq, mach, alfa, reyn = readPlot3d(path: str, 
        >>>         fname_grid='plot3d_grid.xyz', fname_sol='plot3d_sol.bin', binary=True)

        ### Input:
        ```text
        path:       folder that contains the output files
        fname_grid: grid file name
        fname_sol:  solution file name
        binary:     binary or ASCII format
        ```

        ### Return:
        ```text
        xyz:    list of ndarray [ni,nj,nk,3], or None
        qq:     list of ndarray [ni,nj,nk,5], or None
                non-dimensionalized RHO, RHO-U, RHO-V, RHO-W, E
                q1: density by the reference density, rho
                q*: velocity by the reference speed of sound, ar
                q5: total energy per unit volume by rho*ar^2
        mach:   freestream Mach number, = ur/ar
                ur: reference velocity
        alfa:   freestream angle-of-attack
        reyn:   freestream Reynolds number, = rho*ar*Lr/miur
                Lr: reference length
                miur: reference viscosity
        ```
        '''
        xyz = None
        qq = None
        mach = 0.0
        alfa = 0.0
        reyn = 0.0

        if _double_precision:
            r_format = 8
            s_format = 'd'
        else:
            r_format = 4
            s_format = 'f'

        if binary:

            with open(os.path.join(path, fname_grid), 'rb') as f:

                num_block, = st.unpack('i', f.read(4))

                xyz = []
                ni = np.zeros(num_block)
                nj = np.zeros(num_block)
                nk = np.zeros(num_block)

                for n in range(num_block):
                    ni[n],nj[n],nk[n], = st.unpack('iii', f.read(4))

                for n in range(num_block):
                    temp = np.zeros((ni[n],nj[n],nk[n],3))
                    for d in range(3):
                        for k in range(nk[n]):
                            for j in range(nj[n]):
                                for i in range(ni[n]):
                                    temp[i,j,k,d], = st.unpack(s_format, f.read(r_format))

                    xyz.append(copy.deepcopy(temp))

            with open(os.path.join(path, fname_sol), 'r') as f:
                
                num_block, = st.unpack('i', f.read(4))
                qq = []
                ni = np.zeros(num_block)
                nj = np.zeros(num_block)
                nk = np.zeros(num_block)

                for n in range(num_block):
                    ni[n],nj[n],nk[n], = st.unpack('iii', f.read(4))

                for n in range(num_block):
                    temp = np.zeros((ni[n],nj[n],nk[n],5))

                    mach, = st.unpack(s_format, f.read(r_format))   # freestream Mach number
                    alfa, = st.unpack(s_format, f.read(r_format))   # freestream angle-of-attack
                    reyn, = st.unpack(s_format, f.read(r_format))   # freestream Reynolds number
                    time, = st.unpack(s_format, f.read(r_format))   # time

                    for d in range(5):
                        for k in range(nk[n]):
                            for j in range(nj[n]):
                                for i in range(ni[n]):
                                    temp[i,j,k,d], = st.unpack(s_format, f.read(r_format))

                    qq.append(copy.deepcopy(temp))


        else:

            with open(os.path.join(path, fname_grid), 'r') as f:
                xyz = []
                lines = f.readlines()

                line = lines[0].split()
                num_block = int(line[0])
                ni = np.zeros(num_block)
                nj = np.zeros(num_block)
                nk = np.zeros(num_block)

                for n in range(num_block):
                    line = lines[1+n].split()
                    ni[n] = int(line[0])
                    nj[n] = int(line[1])
                    nk[n] = int(line[2])

                k_line = 1+num_block
                k_item = 0
                line   = lines[k_line].split()
                len_line = len(line)
                data = [float(a) for a in line]

                for n in range(num_block):
                    temp = np.zeros((ni[n],nj[n],nk[n],3))
                    for d in range(3):
                        for k in range(nk[n]):
                            for j in range(nj[n]):
                                for i in range(ni[n]):
                                    # Read next line
                                    if k_item >= len_line:
                                        k_line += 1
                                        k_item = 0
                                        line = lines[k_line].split()
                                        len_line = len(line)
                                        data = [float(a) for a in line]

                                    # Assign to xx, yy
                                    temp[i,j,k,d] = data[k_item]
                                    k_item += 1

                    xyz.append(copy.deepcopy(temp))

            with open(os.path.join(path, fname_sol), 'r') as f:
                qq = []
                lines = f.readlines()

                num_block = int(line[0])
                ni = np.zeros(num_block)
                nj = np.zeros(num_block)
                nk = np.zeros(num_block)

                for n in range(num_block):
                    line = lines[1+n].split()
                    ni[n] = int(line[0])
                    nj[n] = int(line[1])
                    nk[n] = int(line[2])

                k_line = 1+num_block
                k_item = 0
                line   = lines[k_line].split()
                len_line = len(line)
                data = [float(a) for a in line]

                for n in range(num_block):
                    temp = np.zeros((ni[n],nj[n],nk[n],5))

                    line = lines[k_line].split()
                    mach = float(line[0])   # freestream Mach number
                    alfa = float(line[1])   # freestream angle-of-attack
                    reyn = float(line[2])   # freestream Reynolds number
                    time = float(line[3])   # time

                    k_line += 1
                    line   = lines[k_line].split()
                    len_line = len(line)
                    data = [float(a) for a in line]

                    for d in range(5):
                        for k in range(nk[n]):
                            for j in range(nj[n]):
                                for i in range(ni[n]):
                                    # Read next line
                                    if k_item >= len_line:
                                        k_line += 1
                                        k_item = 0
                                        line = lines[k_line].split()
                                        len_line = len(line)
                                        data = [float(a) for a in line]

                                    # Assign to xx, yy
                                    temp[i,j,k,d] = data[k_item]
                                    k_item += 1

                    qq.append(copy.deepcopy(temp))

        return xyz, qq, mach, alfa, reyn

    @staticmethod
    def analysePlot3d(Mr: float, qq: np.ndarray, iVar: list, gamma_r=1.4):
        '''
        Calculate fluid variables from plot3d.

        All parameters are non-dimensional.

        >>> var = analysePlot3d(Mr: float, qq, iVar:list, gamma_r=1.4)

        ### Inputs:
        ```text
        Mr:     freestream Mach number
        qq:     ndarray [ni,nj,nk,5] or [ni,nj,4]
        iVar:   list of int, index of variable(s)
        ```

        ### Return:
        ```text
        var:    ndarray [ni,nj,nk,d] or [ni,nj,d]
        ```

        ### Formulas (Index of variable):
        ```text
        dimensionless gas constant:         R  = 1/gamma_r/Mr^2

        1   static density:                 r  = q1
        2   u velocity:                     u  = q2/r/Mr
        3   v velocity:                     v  = q3/r/Mr
        4   w velocity:                     w  = q4/r/Mr
        5   total energy per unit volume:   e  = q5/Mr^2
        6   velocity magnitude:             V  = sqrt(u^2+v^2+w^2)
        7   static temperature:             T  = (gamma_r-1)/R*(e/r-V^2/2)
        8   speed of sound:                 a  = sqrt(gamma_r*R*T)
        9   Mach number:                    M  = V/a
        10  static pressure:                p  = r*T
        11  static pressure coefficient:    cp = 2*R*(p-1)
        12  internal energy:                ei = R*T/(gamma_r-1)
        13  kinetic energy:                 ek = V^2/2
        14  static enthalpy:                h  = gamma_r*R*T/(gamma_r-1)

        15  total energy                    et = e/r
        16  total temperature:              Tt = T*(1+(gamma_r-1)/2/M^2)
        17  total density:                  rt = r*(1+(gamma_r-1)/2/M^2)^(1/(gamma_r-1))
        18  total pressure:                 pt = p*(1+(gamma_r-1)/2/M^2)^(gamma_r/(gamma_r-1))
                                            pt0= (1+(gamma_r-1)/2*M^2)^(gamma_r/(gamma_r-1))
        19  total pressure coefficient:     cpt= 2*R*(pt-pt0)
        20  total enthalpy:                 ht = gamma_r*R*Tt/(gamma_r-1)
        ```
        '''
        if len(qq.shape)==3:
            q = np.expand_dims(qq, 2)           # [ni,nj,1,4]
            q = np.insert(q, 3, 0.0, axis=3)    # [ni,nj,1,5]
        else:
            q = qq

        i_max = np.max(iVar)
        R  = 1/gamma_r/Mr**2

        r  = q[:,:,:,0]
        u  = q[:,:,:,1]/r/Mr
        v  = q[:,:,:,2]/r/Mr
        w  = q[:,:,:,3]/r/Mr
        e  = q[:,:,:,4]/Mr**2
        
        if i_max >= 5:
            V  = np.sqrt(u**2+v**2+w**2)
            T  = (gamma_r-1)/R*(e/r-V**2/2)
            a  = np.sqrt(gamma_r*R*T)
            M  = V/a
            p  = r*T
            cp = 2*R*(p-1)

        var = []
        
        if True:

            if 1 in iVar:
                var.append(r)
            
            if 2 in iVar:
                var.append(u)

            if 3 in iVar:
                var.append(v)

            if 4 in iVar:
                var.append(w)

            if 5 in iVar:
                var.append(e)

            if 6 in iVar:
                var.append(V)
            
            if 7 in iVar:
                var.append(T)

            if 8 in iVar:
                var.append(a)

            if 9 in iVar:
                var.append(M)

            if 10 in iVar:
                var.append(p)

            if 11 in iVar:
                var.append(cp)
            
            if 12 in iVar:
                ei = R*T/(gamma_r-1)
                var.append(ei)

            if 13 in iVar:
                ek = V^2/2
                var.append(ek)

            if 14 in iVar:
                h  = gamma_r*R*T/(gamma_r-1)
                var.append(h)

            if 15 in iVar:
                et = e/r
                var.append(et)

            if 16 in iVar:
                Tt = T*(1+(gamma_r-1)/2/M**2)
                var.append(Tt)
            
            if 17 in iVar:
                rt = r*(1+(gamma_r-1)/2/M**2)**(1/(gamma_r-1))
                var.append(rt)

            if 18 in iVar:
                pt = p*(1+(gamma_r-1)/2/M**2)**(gamma_r/(gamma_r-1))
                var.append(pt)

            if 19 in iVar:
                pt0= (1+(gamma_r-1)/2*M**2)**(gamma_r/(gamma_r-1))
                cpt= 2*R*(pt-pt0)
                var.append(cpt)

            if 20 in iVar:
                ht = gamma_r*R*Tt/(gamma_r-1)
                var.append(ht)

        var = np.array(var)
        if len(qq.shape)==3:
            var = np.squeeze(var, axis=3)
            var = np.transpose(var, axes=(1,2,0))
        else:
            var = np.transpose(var, axes=(1,2,3,0))

        return var

    @staticmethod
    def readsurf2d(path: str, fname_grid='surf.g', fname_sol='surf.q', fname_nam='surf.nam', binary=True, _double_precision=False):
        '''
        Plot3D Format grid and solution for surface data (for CFL3D v6.8):

        >>> xyz, qq = readPlot3d(path: str, 
        >>>         fname_grid='surf.g', fname_sol='surf.q', binary=True)

        ### Input:
        ```text
        path:       folder that contains the output files
        fname_grid: grid file name
        fname_sol:  solution file name
        binary:     binary or ASCII format
        ```

        ### Return:
        ```text
        xyz:    list of ndarray [ni,nj,nk,3], or None
        qq:     list of ndarray [ni,nj,nk,5], or None
                non-dimensionalized RHO, RHO-U, RHO-V, RHO-W, E
                q1: density by the reference density, rho
                q*: velocity by the reference speed of sound, ar
                q5: total energy per unit volume by rho*ar^2
        ```
        '''
        if _double_precision:
            r_format = 8
            s_format = 'd'
        else:
            r_format = 4
            s_format = 'f'
        
        xy = []
        qq = []
        var_list = []

        # if fname_nam is not None:
        #     with open(os.path.join(path, fname_nam), 'r') as f:
        #         lines = f.readlines()
        #     nv = len(lines)
        #     for line in lines:
        #         var_list.append(line.split()[0])
        # else:
        #     nv = 14

        if binary:

            with open(os.path.join(path, fname_grid), 'rb') as f:

                _,num_block,_, = st.unpack('iii', f.read(12))

                _, = st.unpack('i', f.read(4))
                ni = np.zeros(num_block, dtype=np.int32)
                nj = np.zeros(num_block, dtype=np.int32)
                nk = np.zeros(num_block, dtype=np.int32)

                for n in range(num_block):
                    ni[n],nj[n],nk[n], = st.unpack('iii', f.read(12))
                    # print(ni[n], nj[n], nk[n])
                _, = st.unpack('i', f.read(4))

                for n in range(num_block):
                    _, = st.unpack('i', f.read(4))
                    temp  = np.zeros((ni[n],nj[n],nk[n],3))
                    # print(temp.shape)
                    for v in range(3):
                        for k in range(nk[n]):
                            for j in range(nj[n]):
                                for i in range(ni[n]):
                                    temp[i,j,k,v], = st.unpack(s_format, f.read(r_format))
                    # print(temp.shape, temp[:, 0, 0, 2])
                    # print(temp.shape, temp[:, -1, -1, 2])
                    xy.append(copy.deepcopy(temp))
                    _, = st.unpack('i', f.read(4))

            with open(os.path.join(path, fname_sol), 'rb') as f:
                
                _,num_block,_, = st.unpack('iii', f.read(12))

                ni = np.zeros(num_block, dtype=np.int32)
                nj = np.zeros(num_block, dtype=np.int32)
                nk = np.zeros(num_block, dtype=np.int32)
                nv = np.zeros(num_block, dtype=np.int32)

                _, = st.unpack('i', f.read(4))
                for n in range(num_block):
                    ni[n],nj[n],nk[n],nv[n], = st.unpack('iiii', f.read(16))
                _, = st.unpack('i', f.read(4))

                for n in range(num_block):
                    _, = st.unpack('i', f.read(4))
                    temp = np.zeros((ni[n],nj[n],nk[n],nv[n]))

                    # mach, = st.unpack(s_format, f.read(r_format))   # freestream Mach number
                    # alfa, = st.unpack(s_format, f.read(r_format))   # freestream angle-of-attack
                    # reyn, = st.unpack(s_format, f.read(r_format))   # freestream Reynolds number
                    # time, = st.unpack(s_format, f.read(r_format))   # time

                    for d in range(nv[n]):
                        for k in range(nk[n]):
                            for j in range(nj[n]):
                                for i in range(ni[n]):
                                    temp[i,j,k,d], = st.unpack(s_format, f.read(r_format))

                    qq.append(copy.deepcopy(temp))
                    _, = st.unpack('i', f.read(4))

        else:
            raise NotImplementedError('read ASCII file of surface not implemented')
        
        return xy, qq #, var_list

    @staticmethod
    def outputTecplot(xyz, variables, var_name: list, fname='flow-field.dat', append=False):
        '''
        Output tecplot format field data.

        >>> outputTecplot(xyz, variables, var_name: list, fname='flow-field.dat', append=False)

        ### Inputs:
        ```text
        xyz:        ndarray [ni,nj,nk,3] or [ni,nj,3]
        variables:  ndarray [ni,nj,nk,q] or [ni,nj,q]
        ```
        '''
        if len(xyz.shape)==3:
            l2d = True
        else:
            l2d = False
            nk = xyz.shape[2]
            
        ni = xyz.shape[0]
        nj = xyz.shape[1]
        nq = variables.shape[-1]

        if append:
            f = open(fname, 'a')
        else:
            f = open(fname, 'w')

            if l2d:
                f.write('Variables=         X                   Y')
            else:
                f.write('Variables=         X                   Y                   Z')

            for name in var_name:
                f.write('  %18s'%(name))
            f.write('\n')

        if l2d:
            f.write('zone i=%d j=%d \n'%(ni,nj))

            for j in range(nj):
                for i in range(ni):
                    f.write(' %19.12e %19.12e'%(xyz[i,j,0], xyz[i,j,1]))
                    for q in range(nq):
                        f.write(' %19.12e'%(variables[i,j,q]))
                    f.write('\n')

        else:
            f.write('zone i=%d j=%d k=%d \n'%(ni,nj,nk))

            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        f.write(' %19.12e %19.12e %19.12e'%(xyz[i,j,k,0], xyz[i,j,k,1], xyz[i,j,k,2]))
                        for q in range(nq):
                            f.write(' %19.12e'%(variables[i,j,k,q]))
                        f.write('\n')
        
        f.write('\n')
        f.close()


