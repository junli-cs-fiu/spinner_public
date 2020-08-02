# Spinner

This repository contains the code used for the following papers:

Xiaodi Fan, Pedro Soto, Xiaomei Zhong, Dan Xi, Yan Wang, Jun Li, “Leveraging Stragglers in Coded Computing with Heterogeneous Servers”, in Proc. IEEE/ACM 28th International Symposium on Quality of Service (IWQoS), Hangzhou, China, June 15-17, 2020.

# Cloud environment setup

Use Azure's Virtual Machine Scale Sets to create multiple servers at the same time and make sure they can SSH without password with each other and every machine needs to have the required  dependencies.

# Required dependencies

Execute the following commands to install the dependencies successfully.

sudo apt-get update

sudo apt upgrade

sudo apt-get install python3

sudo apt-get install build-essential

install openmpi:
http://edu.itp.phys.ethz.ch/hs12/programming_techniques/openmpi.pdf

sudo apt-get install python3-pip

sudo pip3 install numpy

sudo pip3 install pandas

sudo pip3 install mpi4py

# Command to run the code

mpirun -np 25 --hostfile hosts python mult.py 24 20 960 50000 960 RS 2 name

This is the command to run the experiment when the total number of servers = 25 (1 master, 24 workers), n = 24, k = 20, matrix1 size = 960 * 50000, matrix2 size = 50000 * 960, coding scheme is RS code, nof4 = 2, name = name.

For the coding scheme, you have the option of RS, SP1, SP2, GLO

To run the command successfully, you need to make sure every master and worker have the files needed. 

After that, You will need to place the master's IP address at the first line of the 'hosts' file, and place the workers' IP addresses line by line after the master.

At last, execute the command in the master.
