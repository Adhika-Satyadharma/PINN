import numpy as np
import os

def Write_Case_Setup(CD, NN_Neurons, N, Rand_Seed):
    # In Input Ouytput Equation there is a case spesific variabel

    File_Name = "Reports/Case_Details.txt"
    Temp_Data_File = open(File_Name, "w+")

    # File Name
    Temp_Data_File.writelines("------------------------------------------------------\n")
    Temp_Data_File.writelines("Case Name: \t" + CD.Case_Name +"\n")
    Temp_Data_File.writelines("------------------------------------------------------\n")
    Temp_Data_File.writelines("\n")

    # IO + Equations
    Temp_Data_File.writelines("------------------------------------------------------\n")
    Temp_Data_File.writelines("Input Output Equations\n")
    Temp_Data_File.writelines("------------------------------------------------------\n")
    Temp_Data_File.writelines("Inputs(" + str(CD.n_Input) + "):\t\t" + str(CD.Input_Names) + "\n")
    Temp_Data_File.writelines("Outputs(" + str(CD.n_Output) + "):\t\t" + str(CD.Output_Names) + "\n")
    Temp_Data_File.writelines("Equations(" + str(CD.n_Eq) + "):\t\t" + str(CD.Residual_Names) + "\n")
    Temp_Data_File.writelines("Analytic Exist:\t" + str(CD.Analytic_Exist) + "\n")
    Temp_Data_File.writelines("\n")
    Temp_Data_File.writelines("Domain:\n")
    for i in range(CD.n_Input):
        Temp_Data_File.writelines("\t" + CD.Input_Names[i] + ":\t" + str(CD.lb[i].numpy()) + "\t" + str(CD.ub[i].numpy()) + "\n")
    Temp_Data_File.writelines("\n")
    Temp_Data_File.writelines("Constants:\n")
    Temp_Data_File.writelines("\tC0:\t" + str(CD.rho) + "\n")
    Temp_Data_File.writelines("\tC1:\t" + str(CD.mew) + "\n")
    Temp_Data_File.writelines("\n")
    Temp_Data_File.writelines("\n")

    # NN
    Temp_Data_File.writelines("------------------------------------------------------\n")
    Temp_Data_File.writelines("Neural Network\n")
    Temp_Data_File.writelines("------------------------------------------------------\n")
    Temp_Data_File.writelines("Layers:\t"  + str(len(NN_Neurons)) + "\n")
    Temp_Data_File.writelines("Neurons:\t" + str(NN_Neurons[0]) + "\t" + "(/layer)" + "\n")
    Temp_Data_File.writelines("Total_Epoch:\t" + str(N) + "\n")
    Temp_Data_File.writelines("Seed:\t\t" + str(Rand_Seed) + "\n")
    Temp_Data_File.writelines("Initial Points:\t" + str(CD.N_Points[0]) + "\n")
    Temp_Data_File.writelines("Boundary Points:\t" + str(CD.N_Points[1]) + "\n")
    Temp_Data_File.writelines("Collocation Points:\t" + str(CD.N_Points[2]) + "\n")

    Temp_Data_File.close()

def Write_Array_Data(File_Name, Main_Data, Array_Dim, Data_Type):
    File_Name = File_Name + ".txt"
    Temp_Data_File = open(File_Name, "w+")
    Precission = 20

    if (Array_Dim == 1):
        if Data_Type=="TF":
            for Data_Index in range(len(Main_Data)):
                str_total = str(Main_Data[Data_Index][0])[:Precission]
        elif Data_Type=="NP":
            for Data_Index in range(len(Main_Data)):
                str_total = str(Main_Data[Data_Index])[:Precission]
        Temp_Data_File.writelines(str_total + "\n")

    elif (Array_Dim == 2):
        for Data_Index in range(len(Main_Data)):
            str_total = str(Main_Data[Data_Index][0])[:Precission]
            if (len(Main_Data[Data_Index]) > 1):
                for Variation_Index in range(1, len(Main_Data[Data_Index])):
                    str_total = str_total + " \t" + str(Main_Data[Data_Index][Variation_Index])[:Precission]
            Temp_Data_File.writelines(str_total + "\n")
    Temp_Data_File.close()

def Create_Dir():
    Main_Folder = 'Reports/'
    Folder_Names = ['Data', 'Residual', 'NN']

    Main_Folder_Exist = os.path.isdir(Main_Folder)
    if not(Main_Folder_Exist):
        os.mkdir('Reports/')

    for i in range(len(Folder_Names)):
        Long_Name = Main_Folder + Folder_Names[i] + '/'
        Dir_Exist = os.path.isdir(Long_Name)
        if not(Dir_Exist):
            os.mkdir(Long_Name)

