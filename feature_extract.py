import re
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
dssp_dict = returnSeqdic("dssp_file.txt")
onehot_dict = {'H':[1,0,0,0,0,0,0,0],'B':[0,1,0,0,0,0,0,0],'E':[0,0,1,0,0,0,0,0],'G':[0,0,0,1,0,0,0,0],'I':[0,0,0,0,1,0,0,0],'T':[0,0,0,0,0,1,0,0],'S':[0,0,0,0,0,0,1,0],'-':[0,0,0,0,0,0,0,1]}

def CalDistance(Resdic,Res1,Res2):
    x1,y1,z1 = Resdic[Res1]
    x2,y2,z2 = Resdic[Res2]
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
def getfeatures(targetname):
    features = {}
    #patch features：
    infile = open("protein_name.clusters")
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

    #atomAxs features：
    infile = open("protein_name.atomAxs")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if linea[1] in features.keys():
            if len(features[linea[1]])==1:
                features[linea[1]].append(float(linea[6]))
            else:
                features[linea[1]][1]+=float(linea[6])
    infile.close()
    #axs features：
    infile = open("protein_name.axs")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if linea[1] in features.keys() and len(features[linea[1]])==2:
            features[linea[1]].append(float(linea[5]))
    infile.close()
    #cv features：
    infile = open("protein_name.cv")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if linea[1] in features.keys() and len(features[linea[1]])==3:
            features[linea[1]].append(float(linea[3]))
    infile.close()
    #cvlocal features：
    infile = open("protein_name.cvlocal")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if linea[1] in features.keys() and len(features[linea[1]])==4:
            features[linea[1]].append(float(linea[3]))
    infile.close()
    #pssm & dssp features：
    infile = open("protein_name.pssm")
    infile.readline()
    for line in infile:
        linea = re.split(r"[ ]+",line.strip())
        if len(linea)<10:
            continue
        if linea[0] in features.keys() and len(features[linea[0]])==5:
            features[linea[0]]+=[float(x) for x in linea[2:22]]
            features[linea[0]]+=onehot_dict[dssp_dict[targetname][int(linea[0])-1]]
    infile.close()


    #输出特征值
    outfile = open("protein_name.features",'w')
    for i,j in features.items():
        outfile.write(i)
        for k in j:
            outfile.write('\t'+str(k))
        outfile.write('\n')
    outfile.close()
