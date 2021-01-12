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
def read(path, name, rounding, mode):
    lis = []
    if mode == '.txt':
        with open(path + name + mode) as f:
            for line in f:
                if rounding:
                    lis.append(round(float(line.strip('\n'))))
                else:
                    lis.append(float(line.strip('\n')))
    return lis


'''
split a list into mean list
'''
def means(lis, length):
    means = []
    mean = 0
    for i in range(len(lis)):
        mean += lis[i]
        if i % length == 0 and i != 0:
            means.append(mean // 10)
            mean = 0
    return means