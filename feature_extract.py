

import re
import sys, getopt
#-i input -o ouput -h help
opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
for opt, arg in opts:
      if opt == '-h':
         print('feature_extract.py -i <input protein name> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         TARGETNAME = arg
      elif opt in ("-o", "--ofile"):
         OUTFEATURE = arg

def returnSeqdic(fastapath):
    seqDic = {}
    with open(fastapath) as infile:
        for line in infile:
            if line[0] == '>':
                name = line.strip().split('>')[1]
                seqDic[name] = ""
            else:
                seqDic[name] += line.strip()
    return seqDic
dssp_dict = returnSeqdic("dssp.txt")
onehot_dict = {'H':[1,0,0,0,0,0,0,0],'B':[0,1,0,0,0,0,0,0],'E':[0,0,1,0,0,0,0,0],'G':[0,0,0,1,0,0,0,0],'I':[0,0,0,0,1,0,0,0],'T':[0,0,0,0,0,1,0,0],'S':[0,0,0,0,0,0,1,0],'-':[0,0,0,0,0,0,0,1]}

def getfeatures(targetname,outfeatures):
    features = {}
    #patch：
    infile = open(targetname+".clusters")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+", line.strip())
        if linea[1] in features.keys():
            if float(linea[4])!=0:
                features[linea[1]].append(1)
            else:
                features[linea[1]].append(0)
        else:
            features[linea[1]] = []
            if float(linea[4])!=0:
                features[linea[1]].append(1)
            else:
                features[linea[1]].append(0)
    infile.close()
    #atomAxs：
    infile = open(targetname+".atomAxs")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if linea[1] in features.keys():
            if len(features[linea[1]])==1:
                features[linea[1]].append(float(linea[6]))
            else:
                features[linea[1]][1]+=float(linea[6])
    infile.close()
    #axs：
    infile = open(targetname+".axs")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if linea[1] in features.keys() and len(features[linea[1]])==2:
            features[linea[1]].append(float(linea[5]))
    infile.close()
    #cv：
    infile = open(targetname+".cv")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if linea[1] in features.keys() and len(features[linea[1]])==3:
            features[linea[1]].append(float(linea[3]))
    infile.close()
    #cvlocal：
    infile = open(targetname+".cvlocal")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if linea[1] in features.keys() and len(features[linea[1]])==4:
            features[linea[1]].append(float(linea[3]))
    infile.close()
    #pssm & dssp：
    infile = open(targetname+".pssm")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+", line.strip())
        if len(linea) < 10:
            continue
        if linea[0] in features.keys() and len(features[linea[0]]) == 5:
            features[linea[0]] += [float(x) for x in linea[2:22]]
            features[linea[0]] += onehot_dict[dssp_dict[targetname][int(linea[0]) - 1]]
    infile.close()

    outfile = open(outfeatures+".features",'w')
    for i,j in features.items():
        outfile.write(i)
        for k in j:
            outfile.write('\t'+str(k))
        outfile.write('\n')
    outfile.close()
getfeatures(TARGETNAME,OUTFEATURE)


