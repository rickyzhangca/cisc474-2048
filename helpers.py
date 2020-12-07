'''
save a given list to txt or csv file using w+ policy
'''

def save(path, name, lis, mode):
    file = open(path + name + mode,'w+')
    if mode == '.txt':  
        for i in range(len(lis)):
            file.write(str(lis[i])+"\n")     
        file.close()
    elif mode == '.csv':
        file.write('Episode,Weight\n') ###
        for i in range(lis.shape[0]):
            file.write(str(i) + ',' + str(lis[i][0])+'\n') 
    file.close()
    print(path + name + mode + " is written")


'''
read a csv/txt file to list of int/float
'''
def read(path, name, mode):
    lis = []
    if mode == '.txt':
        with open(path + name + mode) as f:
            for line in f:
                lis.append(round(float(line.strip('\n'))))
    return lis