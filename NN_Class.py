
# Import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize as opt
from matplotlib.colors import Normalize
from time import time
import math


class PINN_NeuralNet(tf.keras.Model):
    def __init__(self, Case_Boundary, NN_IO_Dim, NN_Neurons,
                 activation='tanh',
                 kernel_initializer='glorot_normal',
                 **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = len(NN_Neurons)
        self.output_dim = NN_IO_Dim[1]
        self.lb = Case_Boundary[0]
        self.ub = Case_Boundary[1]

        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0 * (x - self.lb ) /(self.ub - self.lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(NN_Neurons[_],
                                             activation=tf.keras.activations.get(activation),
                                             kernel_initializer=kernel_initializer)
                       for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(self.output_dim)

    def call(self, X):
        Z = self.scale(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)

class PINN_Channel_Flow():
    def __init__(self, model, CD, PR, X_r, Backup_Interval, Special_LF):
        self.model = model
        self.CD = CD
        self.PR = PR
        self.x = X_r[:, 0:1]
        self.y = X_r[:, 1:2]
        self.Backup_Interval = Backup_Interval
        self.DTYPE = CD.DTYPE
        self.Special_LF = Special_LF

        self.hist = []
        self.iter = 0
        self.Backup_Index = 0
        self.Backup_Check = True

    def loss_fn(self, x_bi, BCI_Data):
        r = self.CD.Get_Sim_Param(self.x, self.y, "R")
        r = tf.concat(r, axis=1)
        phi_r = tf.reduce_mean(tf.square(r))
        loss = phi_r

        # x_bi[0] = Dirchlet
        All_Var_Pred = self.model(x_bi[0])
        u_pred = All_Var_Pred[:, 0:1]
        v_pred = All_Var_Pred[:, 1:2]
        loss += tf.reduce_mean(tf.square(BCI_Data[0][0] - u_pred))
        loss += tf.reduce_mean(tf.square(BCI_Data[0][1] - v_pred))

        # x_bi[1] = Neumann
        Temp_x = x_bi[1][:, 0:1]
        Temp_y = x_bi[1][:, 1:2]
        [D1_Pred] = self.CD.Get_Sim_Param(Temp_x, Temp_y, ["D1"])
        loss += tf.reduce_mean(tf.square(BCI_Data[1][0] - D1_Pred[0]))
        loss += tf.reduce_mean(tf.square(BCI_Data[1][1] - D1_Pred[2]))

        # Special
        for i in range(len(self.Special_LF)):
            if self.Special_LF[i] == "max":
                loss += tf.reduce_max(tf.square(r))
            elif self.Special_LF[i] == "Global_Flow_Rate":
                # WARNING = Have to know the average u,v in the sim domain
                LGFR_x = 1./3. - tf.reduce_mean(u_pred)
                #LGFR_y = 0. - tf.reduce_mean(v_pred)
                loss += (LGFR_x * LGFR_x)
                #loss += (LGFR_y * LGFR_y)
        return loss

    def get_grad(self, x_bi, BCI_Data):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(x_bi, BCI_Data)
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape

        return loss, g

    def solve_with_TFoptimizer(self, x_bi, BCI_Data, N, n_Print, Initial_Time, Switch_Param):
        # Init Vars
        self.Total_Iteration = N
        self.Report_Interval = n_Print
        self.Initial_Time = Initial_Time
        Switch_Param = Switch_Param
        Use_Adam = True

        # Clear File
        Progress_File = open("Reports/Progress.txt", "w+")
        Progress_File.close()

        # Main Func
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,1e-4])
        optim = tf.keras.optimizers.Adam(learning_rate=lr)

        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad(x_bi, BCI_Data)
            optim.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss

        while Use_Adam:
            loss = train_step()
            self.current_loss = loss.numpy()
            self.callback()
            if Switch_Param[0] == "iter":
                if self.iter > Switch_Param[1]:
                    Use_Adam = False
                    print("Iter is above " + str(Switch_Param[1]))
            elif Switch_Param[0] == "LS":
                if self.current_loss < Switch_Param[1]:
                    Use_Adam = False
                    print("Loss Function is below " + str(Switch_Param[1]))

        return self.iter

    def solve_with_ScipyOptimizer(self, X, u, method='L-BFGS-B', **kwargs):
        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""

            weight_list = []
            shape_list = []

            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())

            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()

        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape

                # Weight matrices
                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    new_val = tf.reshape(weight_list[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw

                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx + vs[0]]
                    idx += vs[0]

                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1

                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, self.DTYPE))

        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""

            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, grad = self.get_grad(X, u)

            # Store current loss for callback function
            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss

            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())

            # Gradient list to array
            grad_flat = np.array(grad_flat, dtype=np.float64)

            # Return value and gradient of \phi as tuple
            return loss, grad_flat

        return opt.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)

    def callback(self, xr = None):
        if self.iter % self.Report_Interval == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter, self.current_loss))
        self.hist.append(self.current_loss)
        self.iter += 1

        if (self.iter==1):
            self.LS_Value_Check = 10 ** math.floor(math.log10(self.hist[0]))
        else:
            if (self.current_loss < self.LS_Value_Check):
                print("Loss Function is below: " + str(self.LS_Value_Check))
                self.PR.Execute_All(self, "C", str(self.LS_Value_Check))
                self.LS_Value_Check = self.LS_Value_Check/10

        if self.Backup_Check:
            if (self.iter == self.Backup_Interval[self.Backup_Index]):
                self.PR.Execute_All(self, "B", str(self.iter))
                self.Backup_Index += 1
                if (self.Backup_Index==len(self.Backup_Interval)):
                    self.Backup_Check = False



class Print_Results():
    # ----------------------------------------------------------------------------------
    # Initialize (Might need to be adjusted)
    # ----------------------------------------------------------------------------------
    # Only print at t = 0, t = 0.05 and t = 0.1
    def __init__(self, Case_Details, lb, ub, DTYPE):
        self.CD = Case_Details
        self.lb = lb
        self.ub = ub
        self.N = 500
        #self.Time_Interval = [0., 0.05, 0.1]
        #self.U = [0 for x in range(len(self.Time_Interval))]
        #self.V = [0 for x in range(len(self.Time_Interval))]

        xspace = np.linspace(lb[0], ub[0], self.N + 1, dtype=DTYPE)
        yspace = np.linspace(lb[1], ub[1], self.N + 1, dtype=DTYPE)
        self.X, self.Y = np.meshgrid(xspace, yspace)
        self.xy = np.stack([self.X.flatten(), self.Y.flatten()], axis=-1)

        if Case_Details.Analytic_Exist:
            #for Time_Index in range(len(self.Time_Interval)):
                #T = self.X * 0. + self.Time_Interval[Time_Index]
                #[self.U[Time_Index], self.V[Time_Index]] = Case_Details.Calc_Analytic(self.X, self.Y, T)
            self.Analytical_Result = Case_Details.Calc_Analytic(self.X, self.Y)
            for i in range(self.CD.n_Output):
                File_Name = "Analytic_"+ self.CD.Output_Names[i]
                self.Save_Result_Image(self.Analytical_Result[i], self.X, self.Y, "Data", File_Name, self.CD.Output_Names[i])

    def Execute_All(self, NN_Solver, Mode, NN_Save_Name):
        if Mode == "F":
            Save_Name = 'Reports/'+ NN_Save_Name
            NN_Save_Name = "FINAL"
            self.Print_LS_Image(NN_Solver.hist, NN_Save_Name)
        elif Mode == "C":
            # NN_Save_Name = str(self.LS_Value_Check)
            Save_Name = 'Reports/NN/Backup-LS-' + NN_Save_Name
        elif Mode == "B":
            # NN_Save_Name = str(self.iter)
            Save_Name = 'Reports/NN/Backup-iter-' + NN_Save_Name
            self.Print_LS_Image(NN_Solver.hist, NN_Save_Name)

        NN_Solver.model.save(Save_Name)
        self.Calc_Sim_Param()
        if (self.CD.Analytic_Exist):
            self.Calc_Analytical_Error()
        self.Print_Result_Data(NN_Solver.iter, NN_Solver.current_loss, NN_Solver.Initial_Time, Mode)

        if Mode == "F" or Mode == "B":
            self.Print_Result_Image(NN_Save_Name)
    # ----------------------------------------------------------------------------------
    # Loss Function
    # ----------------------------------------------------------------------------------
    def Print_LS_Image(self, hist, File_ID):      # Loss Function
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.semilogy(range(len(hist)), hist, 'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi_{n_{epoch}}$');
        plt.savefig('Reports/Loss_Function-' + File_ID + '.png', bbox_inches='tight', dpi=300);
        plt.close()

    # ----------------------------------------------------------------------------------
    # Calculation (Might need to be adjusted)
    # ----------------------------------------------------------------------------------
    def Calc_Sim_Param(self):
        xy2 = tf.convert_to_tensor(self.xy)
        Temp_x = xy2[:, 0:1]
        Temp_y = xy2[:, 1:2]
        [Coor, self.Main_Var_NN, Derivative1, Derivative2, Temp_Residuals] = self.CD.Get_Sim_Param(Temp_x, Temp_y, "A")

        self.Residuals = []
        for i in range(self.CD.n_Eq):
            self.Residuals.append(Temp_Residuals[i].numpy().reshape(self.X.shape))

    def Calc_Analytical_Error(self):
        self.E_A = []
        for i in range(self.CD.n_Output):
            Temp_Var = self.Main_Var_NN[i].numpy()
            self.E_A.append(self.Analytical_Result[i] - Temp_Var.reshape(self.X.shape))

    def Calc_MinMax(self, Main_Data, Dim):
        Min_Value = 999999
        Max_Value = -999999
        if (Dim == 1):
            for Data_Index in range(len(Main_Data)):
                if Main_Data[Data_Index]>Max_Value:
                    Max_Value=Main_Data[Data_Index]
                if Main_Data[Data_Index]<Min_Value:
                    Min_Value=Main_Data[Data_Index]
        elif (Dim==2):
            for Data_Index1 in range(len(Main_Data)):
                for Data_Index2 in range(len(Main_Data[Data_Index1])):
                    if Main_Data[Data_Index1][Data_Index2]>Max_Value:
                        Max_Value = Main_Data[Data_Index1][Data_Index2]
                    if Main_Data[Data_Index1][Data_Index2]<Min_Value:
                        Min_Value = Main_Data[Data_Index1][Data_Index2]

        return [Min_Value, Max_Value]

    # ----------------------------------------------------------------------------------
    # Printing (Save_Result_Image might need some adjustment)
    # ----------------------------------------------------------------------------------
    def Print_Result_Data(self, iter, LS_Value, Initial_Time, ID):
        File_Holder = open("Reports/Progress.txt", "a+")

        #   ID: (B) Backup  # (L) Level    # (F) Final
        str_total = ID + "\t" + str(iter) + "\t" + str(LS_Value) + "\t" + str(time() - Initial_Time)
        for i in range(self.CD.n_Eq):
            [Min_R, Max_R] = self.Calc_MinMax(self.Residuals[i], 2)
            str_total = str_total + "\t" + str(Min_R) + "\t" + str(Max_R)

        if (self.CD.Analytic_Exist):
            for i in range(self.CD.n_Output):
                [Min_EA, Max_EA] = self.Calc_MinMax(self.E_A[i], 2)
                str_total = str_total + "\t" + str(Min_EA) + "\t" + str(Max_EA)

        File_Holder.writelines(str_total + "\n")
        File_Holder.close()

    def Print_Result_Image(self, File_ID):
        # Default : Print Main Var & Residual
        for i in range (self.CD.n_Output):
            File_Name = self.CD.Output_Names[i]+"_NN-" + File_ID
            self.Save_Result_Image(self.Main_Var_NN[i].numpy().reshape(self.X.shape), self.X, self.Y, "Data", File_Name, self.CD.Output_Names[i])

        for i in range (self.CD.n_Eq):
            File_Name = self.CD.Residual_Names[i]+"-" + File_ID
            self.Save_Result_Image(self.Residuals[i], self.X, self.Y, "Residual", File_Name, self.CD.Residual_Names[i])

        if (self.CD.Analytic_Exist):
            for i in range (self.CD.n_Output):
                File_Name = "Error_to_Analytic-" + self.CD.Output_Names[i] + File_ID
                self.Save_Result_Image(self.E_A[i], self.X, self.Y, "Data", File_Name, "Error")

    def Save_Result_Image(self, Main_Data, X, Y, Folder, File_Name, Axis_Name):
        # Default Axis Follow Case_Details.Input_Names
        fig = plt.figure(figsize=(15, 10))
        vmin, vmax = np.min(np.min(Main_Data)), np.max(np.max(Main_Data))   # <-------- Modify if need costum scale
        plt.pcolormesh(X, Y, Main_Data, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
        font1 = {'family': 'serif', 'size': 20}

        plt.title(File_Name, fontdict=font1)
        plt.xlabel(self.CD.Input_Names[0], fontdict=font1)          # <-------- Could be modified
        plt.ylabel(self.CD.Input_Names[1], fontdict=font1)          # <-------- Could be modified
        plt.tick_params(axis='both', which='major', labelsize=15)

        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label(Axis_Name, fontdict=font1)
        cbar.mappable.set_clim(vmin, vmax)
        cbar.ax.tick_params(labelsize=15)
        plt.savefig('Reports/' + Folder + "/" + File_Name + ".png", bbox_inches='tight', dpi=300);
        plt.close()

