import re
import pickle as pkl
import os
import numpy as np
import sys, getopt

opts, args = getopt.getopt(sys.argv[1:],"hi:o:s:",["ifile=","ofile=","species="])
for opt, arg in opts:
      if opt == '-h':
          print('adjmatrix_extract.py -i <input protein name> -o <outputfile>')
          sys.exit()
      elif opt in ("-i", "--ifile"):
          TARGETNAME = arg
      elif opt in ("-o", "--ofile"):
          OUTADJ = arg
      elif opt in ("-s", "--species"):
          SPECIES = arg


if SPECIES == "human":
    with open("Human_RSA0.2.pkl", 'rb') as infile:
        RSAdict = pkl.load(infile)
elif SPECIES == "yeast":
    with open("Yeast_RSA0.2.pkl", 'rb') as infile:
        RSAdict = pkl.load(infile)
else:
    print("species error!")
    exit()

AAtype = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
          'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

def CalDistance(Resdic,Res1,Res2):
    x1,y1,z1 = Resdic[Res1]
    x2,y2,z2 = Resdic[Res2]
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
def getpdb(targetname,outfile):
    Resdic = {}
    with open(targetname+".pdb") as infile:
        for line in infile:
            if line.startswith('ATOM'):
                linea = re.split(r" +", line.strip())
                Ele = linea[2]
                Name = linea[3]
                Chain = line[21]
                Number = line[22:26].replace(' ', '')
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if Name in AAtype and Ele == 'CA':
                    Resdic[Chain + str(Number)] = [x, y, z]

    jet2_info = np.loadtxt(targetname+".features")
    patch_list = []
    patch_info = jet2_info[:, 1]
    for i in range(len(patch_info)):
        if patch_info[i] == 1:
            patch_list.append('A' + str(i + 1))
    final_patch_list = []
    for i in patch_list:
        for j in Resdic.keys():
            dis = CalDistance(Resdic, i, j)
            if dis < 10:
                final_patch_list.append(j)
    tmp_keys = [int(x[1:]) - 1 for x in list(set(final_patch_list))]
    tmp_keys_plus = [int(x) - 1 for x in RSAdict[targetname]]
    tmp_keys = list(set(tmp_keys + tmp_keys_plus))
    residues = []
    for i in range(len(tmp_keys)):
        residues.append('A' + str(i + 1))
    outfile = open(outfile+".adj", 'w')
    for i in residues:
        for j in residues:
            if CalDistance(Resdic, i, j) < 10:
                outfile.write(i[1:] + '\t' + j[1:] + '\n')
getpdb(TARGETNAME,OUTADJ)
