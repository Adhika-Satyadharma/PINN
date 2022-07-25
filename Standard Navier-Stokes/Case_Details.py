
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
    Case_Name = 'Simplified BFS'
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

        self.Input_Names = ["x", "y"]
        self.Output_Names = ["u", "v", "p"]
        self.Residual_Names = ["Mass_Imbalance", "Mom-X_Imbalance", "Mom-Y_Imbalance"]
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

        u = Main_Var[0]
        v = Main_Var[1]
        p = Main_Var[2]

        u_x = Derivative1[0]
        u_y = Derivative1[1]
        v_x = Derivative1[2]
        v_y = Derivative1[3]
        p_x = Derivative1[4]
        p_y = Derivative1[5]

        u_xx = Derivative2[0]
        u_yy = Derivative2[1]
        v_xx = Derivative2[2]
        v_yy = Derivative2[3]

        nu = self.mew / self.rho

        R1 = self.Mass_Eq(u_x, v_y)
        R2 = self.Mom_x_Eq(u, v, u_x, u_y, p_x, u_xx, u_yy, self.rho, nu)
        R3 = self.Mom_y_Eq(u, v, v_x, v_y, p_y, v_xx, v_yy, self.rho, nu)

        return [R1, R2, R3]

    def Mass_Eq(self, u_x, v_y):
        return u_x + v_y

    def Mom_x_Eq(self, u, v, u_x, u_y, p_x, u_xx, u_yy, rho, nu):
        Convection_Terms = (u * u_x) + (v * u_y)
        Pressure_Terms = -p_x / rho
        Dissipation_Terms = nu * (u_xx + u_yy)
        # Body_Force =
        # Source_Terms =
        return Convection_Terms - Pressure_Terms - Dissipation_Terms

    def Mom_y_Eq(self, u, v, v_x, v_y, p_y, v_xx, v_yy, rho, nu):
        Convection_Terms = u * v_x + v * v_y
        Pressure_Terms = -p_y / rho
        Dissipation_Terms = nu * (v_xx + v_yy)
        # Body_Force =
        # Source_Terms =
        return Convection_Terms - Pressure_Terms - Dissipation_Terms

    def Get_Sim_Param(self, x, y, Filter):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            All_Var = self.model(tf.stack([x[:, 0], y[:, 0]], axis=1))

            # Main Var
            u = All_Var[:, 0:1]
            v = All_Var[:, 1:2]
            p = All_Var[:, 2:3]

            # First Order
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)

            # Second Order
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            v_xx = tape.gradient(v_x, x)
            v_yy = tape.gradient(v_y, y)
        del tape

        Coor = [x, y]
        Main_Var = [u, v, p]
        Derivative1 = [u_x, u_y, v_x, v_y, p_x, p_y]
        Derivative2 = [u_xx, u_yy, v_xx, v_yy]
        Residuals = self.Calc_Residual(Coor, Main_Var, Derivative1, Derivative2)

        Results = []
        for i in range(len(Filter)):
            if (Filter == "A"):     # All
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
                u_D_Temp.append(Main_Var_DA[DA_Index][1])       # Interpolate: p,u,v
                v_D_Temp.append(Main_Var_DA[DA_Index][2])       # Interpolate: p,u,v

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
        X_c = tf.concat([x_c, y_c], axis=1)

        fig = plt.figure(figsize=(12, 9))
        #ax.scatter(x_i, y_i, marker='X')
        #plt.scatter(X_DA[0][:, 0:1], X_DA[0][:, 1:2], marker='X')
        plt.scatter(X_b_D[:, 0:1], X_b_D[:, 1:2], marker='X')
        plt.scatter(X_b_N[:, 0:1], X_b_N[:, 1:2], marker='X')
        plt.scatter(x_c, y_c, c='r', marker='.', alpha=0.1)
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title('Positions of collocation points and boundary data');
        plt.savefig('Reports/Point_Distribution.png', bbox_inches='tight', dpi=300)
        plt.close()

        return [xbi_data, ubi_data, X_c]