
# Import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from NN_Class import*

def Read_Interpoalted_Data_TF(Data_File_Name):
    # This is a function to read a 2D Interpoalte data from Fluent
    # This is optimized for TF

    Temp_Data_File = open(Data_File_Name, "r")
    Temp_All_Lines = Temp_Data_File.readlines()
    Cell_Count = round(float(Temp_All_Lines[2]))
    N_Output = round(float(Temp_All_Lines[3]))

    XY_Loc = [0 for x in range(Cell_Count)]
    Main_Vars_Data = [0 for y in range(N_Output)]

    Starting_Line = 3 + N_Output + 1
    for Line_Index in range(Cell_Count):
        XY_Loc[Line_Index] = [float(Temp_All_Lines[Line_Index + Starting_Line]),
                              float(Temp_All_Lines[Cell_Count + Line_Index + Starting_Line])]
    TF_XY = tf.convert_to_tensor(XY_Loc)

    for i in range(N_Output):
        Temp_Array = []
        for Line_Index in range(Cell_Count):
            Temp_Array.append([float(Temp_All_Lines[(i+2) * Cell_Count + Line_Index + Starting_Line])])
        Temp_TF = tf.convert_to_tensor(Temp_Array)
        Main_Vars_Data[i] = Temp_TF
    Temp_Data_File.close()

    return [TF_XY, Main_Vars_Data]

class Channel_Flow():
    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(self, Domain, Constant_List, N_Points, DA_Names, DTYPE):
        self.xmin = Domain[0]
        self.xmax = Domain[1]
        self.ymin = Domain[2]
        self.ymax = Domain[3]
        self.rho = Constant_List[0]
        self.mew = Constant_List[1]
        self.N_Points = N_Points
        self.DA_Names = DA_Names
        self.DTYPE = DTYPE

        self.Case_Name = 'Simplified BFS'
        self.Input_Names = ["x", "y"]
        self.Output_Names = ["u", "v"]
        self.Residual_Names = ["Mom_Imbalance", "Omega_Imbalance"]
        self.Analytic_Exist = False

        self.n_Input = len(self.Input_Names)
        self.n_Output = len(self.Output_Names)
        self.n_Eq = len(self.Residual_Names)

    def Call_Boundary(self):
        if (self.n_Input==2):
            self.lb = tf.constant([self.xmin, self.ymin], dtype=self.DTYPE)
            self.ub = tf.constant([self.xmax, self.ymax], dtype=self.DTYPE)
        elif (self.n_Input==3):
            self.lb = tf.constant([self.xmin, self.ymin, self.tmin], dtype=self.DTYPE)
            self.ub = tf.constant([self.xmax, self.ymax, self.tmax], dtype=self.DTYPE)
        elif (self.n_Input==4):
            self.lb = tf.constant([self.xmin, self.ymin, self.zmin, self.tmin], dtype=self.DTYPE)
            self.ub = tf.constant([self.xmax, self.ymax, self.zmax, self.tmax], dtype=self.DTYPE)
        return [self.lb, self.ub]

    def Call_Case_Dim(self):
        return [self.n_Input, self.n_Output]

    def Set_Model(self, model):
        self.model = model
    # ----------------------------------------------------------------------------------
    # BC, IC, Residual, Analytic
    # ----------------------------------------------------------------------------------
    def Calc_BC(self, x, y, Code):
        if (Code == "Inlet"):
            # Parabolic Velocity profile
            f = 2. * y - 1.
            f2 = f * f
            u = 1. - f2
            v = 0. * y
        elif(Code == "Outlet"):
            # Neumann BC
            u = 0. * y  # du/dx
            v = 0. * y  # dv/dx
        elif(Code == "Wall"):
            u = 0. * y
            v = 0. * y
        return [u, v]

    def Calc_IC(self, x, y):
        # Dummy
        u = x + y
        v = x - y
        return [u, v]

    def Calc_Analytic(self, x, y):
        # Dummy
        u = 1. - y * y
        v = 0. * y
        p = -2. * self.mew*x
        return [u, v, p]

    def Calc_Residual(self, Coor, Main_Var, Derivative1, Derivative2):
        x = Coor[0]
        y = Coor[1]

        psi = Main_Var[0]
        omega = Main_Var[1]

        psi_x = Derivative1[0]
        psi_y = Derivative1[1]
        omega_x = Derivative1[2]
        omega_y = Derivative1[3]

        psi_xx = Derivative2[0]
        psi_yx = Derivative2[1]
        psi_yy = Derivative2[2]
        omega_xx = Derivative2[3]
        omega_yy = Derivative2[4]

        nu = self.mew / self.rho

        R1 = self.Mom_Eq(psi_x, psi_y, omega_x, omega_y, omega_xx, omega_yy, nu)
        R2 = self.Conv_Eq(omega, psi_xx, psi_yy)

        return [R1, R2]

    def Mom_Eq(self, psi_x, psi_y, omega_x, omega_y, omega_xx, omega_yy, nu):
        Convection_Terms = (psi_y * omega_x) - (psi_x * omega_y)
        Dissipation_Terms = nu * (omega_xx + omega_yy)
        # Body_Force =
        # Source_Terms =
        return Convection_Terms - Dissipation_Terms

    def Conv_Eq(self, omega, psi_xx, psi_yy):
        return omega + psi_xx + psi_yy

    def Get_Sim_Param(self, x, y, Filter):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            All_Var = self.model(tf.stack([x[:, 0], y[:, 0]], axis=1))

            # Main Var
            psi = All_Var[:, 0:1]
            omega = All_Var[:, 1:2]

            # First Order
            psi_x = tape.gradient(psi, x)  # -v
            psi_y = tape.gradient(psi, y)  # u
            omega_x = tape.gradient(omega, x)
            omega_y = tape.gradient(omega, y)

            # Second Order
            psi_xx = tape.gradient(psi_x, x)  # -dv/dx
            psi_yx = tape.gradient(psi_y, x)  # du/dx
            psi_yy = tape.gradient(psi_y, y)  # du/dy
            omega_xx = tape.gradient(omega_x, x)
            omega_yy = tape.gradient(omega_y, y)
        del tape

        Coor = [x, y]
        Main_Var = [psi, omega]
        Derivative1 = [psi_x, psi_y, omega_x, omega_y]
        Derivative2 = [psi_xx, psi_yx, psi_yy, omega_xx, omega_yy]
        Residuals = self.Calc_Residual(Coor, Main_Var, Derivative1, Derivative2)

        Results = []
        for i in range(len(Filter)):
            if (Filter == "A"):  # All
                Results = [Coor, Main_Var, Derivative1, Derivative2, Residuals]
            elif (Filter[i] == "R"):
                Results.append(Residuals)
            elif (Filter[i] == "C"):
                Results.append(Coor)
            elif (Filter[i] == "M"):
                Results.append(Main_Var)
            elif (Filter[i] == "D1"):
                Results.append(Derivative1)
            elif (Filter[i] == "D2"):
                Results.append(Derivative2)

        return Results
    # ----------------------------------------------------------------------------------
    # Points
    # ----------------------------------------------------------------------------------
    def Gen_Points_Initial(self):
        N_i = self.N_Points[0]
        x_i = tf.random.uniform((N_i, 1), self.lb[0], self.ub[0], dtype=self.DTYPE)
        y_i = tf.random.uniform((N_i, 1), self.lb[1], self.ub[1], dtype=self.DTYPE)
        t_i = tf.zeros((N_i, 1), dtype=self.DTYPE)
        X_i = tf.concat([x_i, y_i, t_i], axis=1)
        [u_i, v_i] = self.Calc_IC(x_i, y_i, t_i)

        return [X_i, u_i, v_i]

    def Gen_Points_Boundary(self):
        Total_Surface = 5
        N_b = int(self.N_Points[1]/Total_Surface)

        # Coor
        B_xLb = [self.lb[0], self.ub[0], self.lb[0], self.lb[0], self.lb[0]]
        B_xUb = [self.ub[0], self.ub[0], self.ub[0], self.lb[0], self.lb[0]]
        B_yLb = [self.ub[1], self.lb[1], self.lb[1], self.lb[1], 0.        ]
        B_yUb = [self.ub[1], self.ub[1], self.lb[1], 0.        , self.ub[1]]

        # Type
        BC_Type1 = ["D", "N", "D", "D", "D"]                        # Dirichlet or Neumann
        BC_Type2 = ["Wall", "Outlet", "Wall", "Wall", "Inlet"]      # See BC
        BC_Type3 = ["U", "U", "U", "U", "U"]                        # Uniform or Nonuniform

        # Generate Points
        x_b = []
        y_b = []
        for i in range(Total_Surface):
            x_b.append(tf.random.uniform((N_b, 1), B_xLb[i], B_xUb[i], dtype=self.DTYPE))
            y_b.append(tf.random.uniform((N_b, 1), B_yLb[i], B_yUb[i], dtype=self.DTYPE))

        # Calc BC Values
        u_b = []
        v_b = []
        for i in range(Total_Surface):
            [u_Temp, v_Temp] = self.Calc_BC(x_b[i], y_b[i], BC_Type2[i])
            u_b.append(u_Temp)
            v_b.append((v_Temp))

        # Filter
        D_Data = [[] for x in range(4)]     # x,y,u,v
        N_Data = [[] for x in range(4)]     # x,y,u,v
        for i in range(Total_Surface):
            if (BC_Type1[i] == "D"):
                D_Data[0].append(x_b[i])
                D_Data[1].append(y_b[i])
                D_Data[2].append(u_b[i])
                D_Data[3].append(v_b[i])
            elif (BC_Type1[i] == "N"):
                N_Data[0].append(x_b[i])
                N_Data[1].append(y_b[i])
                N_Data[2].append(u_b[i])
                N_Data[3].append(v_b[i])

        # Combine
        Concat_D = [[] for x in range(4)]
        Concat_N = [[] for x in range(4)]
        for i in range(4):
            Concat_D[i] = tf.concat(D_Data[i], 0)
            Concat_N[i] = tf.concat(N_Data[i], 0)

        X_b_D = tf.concat([Concat_D[0], Concat_D[1]], axis=1)
        X_b_N = tf.concat([Concat_N[0], Concat_N[1]], axis=1)

        return [X_b_D, Concat_D[2], Concat_D[3], X_b_N, Concat_N[2], Concat_N[3]]

    def Generate_Points(self):
        # Initial
        if (self.N_Points[0] > 0):
            [X_i, u_i, v_i] = self.Gen_Points_Initial()

        # Boundary
        if (self.N_Points[1] > 0):
            [X_b_D, u_b_D, v_b_D, X_b_N, u_b_N, v_b_N] = self.Gen_Points_Boundary()

        # Data Assimilation
        if len(self.DA_Names)>0:
            X_DA = []
            Main_Var_DA = []
            for DA_Index in range(len(self.DA_Names)):
                [X_Temp, MV_Temp] = Read_Interpoalted_Data_TF(self.DA_Names[DA_Index])
                X_DA.append(X_Temp)
                Main_Var_DA.append(MV_Temp)

        # Combine Dirichlet
        x_D_Temp = []
        u_D_Temp = []
        v_D_Temp = []
        if (self.N_Points[0] > 0):
            x_D_Temp.append(X_i)
            u_D_Temp.append(u_i)
            v_D_Temp.append(v_i)
        if (self.N_Points[1] > 0):
            x_D_Temp.append(X_b_D)
            u_D_Temp.append(u_b_D)
            v_D_Temp.append(v_b_D)
        if len(self.DA_Names)>0:
            for DA_Index in range(len(self.DA_Names)):
                x_D_Temp.append(X_DA[DA_Index])
                u_D_Temp.append(Main_Var_DA[DA_Index][0])
                v_D_Temp.append(Main_Var_DA[DA_Index][1])

        X_D_Total = tf.concat(x_D_Temp, 0)
        u_D_Total = tf.concat(u_D_Temp, 0)
        v_D_Total = tf.concat(v_D_Temp, 0)
        xbi_data = [X_D_Total, X_b_N]
        ubi_data = [[u_D_Total, v_D_Total], [u_b_N, v_b_N]]


        # Collocation
        N_c = self.N_Points[2]
        #x_c = tf.random.uniform((N_c, 1), self.lb[0], self.ub[0], dtype=self.DTYPE)
        Dummy = tf.random.uniform((N_c, 1), 0., 1., dtype=self.DTYPE)
        x_c = tf.cos(Dummy * np.pi /2.) * (self.lb[0] - self.ub[0]) + self.ub[0]
        y_c = tf.random.uniform((N_c, 1), self.lb[1], self.ub[1], dtype=self.DTYPE)

        # C2
        #Dummy1 = tf.random.uniform((N_c, 1), 0., 1., dtype=self.DTYPE)
        #Dummy2 = tf.random.uniform((N_c, 1), 0., 1., dtype=self.DTYPE)
        #x_c2 = tf.cos(Dummy1 * np.pi /2.) * (self.lb[0] - 5.) + 5.
        #y_c2 = tf.cos(Dummy2 * np.pi /2.) * (-1. - 0.) + 0.

        #x_c = tf.concat([x_c1, x_c2], 0)
        #y_c = tf.concat([y_c1, y_c2], 0)
        X_c = tf.concat([x_c, y_c], axis=1)

        fig = plt.figure(figsize=(12, 9))
        #ax.scatter(x_i, y_i, marker='X')
        plt.scatter(X_b_D[:, 0:1], X_b_D[:, 1:2], marker='X')
        plt.scatter(X_b_N[:, 0:1], X_b_N[:, 1:2], marker='X')
        plt.scatter(x_c, y_c, c='r', marker='.', alpha=0.1)
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title('Positions of collocation points and boundary data');
        plt.savefig('Reports/Point_Distribution.png', bbox_inches='tight', dpi=300)
        plt.close()

        return [xbi_data, ubi_data, X_c]

    def Add_Collocation_Points(self, Prev_X_c, Main_Coor, radius):
        # Add N points to Prev_X_c @ Main_Coor +- radius
        # Shape = Box (for simplicity)
        N = 100             # Set as default, subject to change
        x_c_add = tf.random.uniform((N, 1), Main_Coor[0] - radius, Main_Coor[0] + radius, dtype=self.DTYPE)
        y_c_add = tf.random.uniform((N, 1), Main_Coor[1] - radius, Main_Coor[1] + radius, dtype=self.DTYPE)
        X_c_add = tf.concat([x_c_add, y_c_add], axis=1)

        New_X_c = tf.concat([Prev_X_c, X_c_add], axis=0)

        return New_X_c