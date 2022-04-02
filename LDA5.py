import numpy as np
import pandas as pd
import math
import datetime
for p in range(1):
    dep=0.02+p*0.05
    Ca=np.zeros([5825,1])+30
    Ce=np.zeros([339,1])+30
    train = pd.read_csv( '0.02_d_rt_train.txt', sep='\t',float_precision='3')
    userID = np.array(train.iloc[:, 0])
    serviceID = np.array(train.iloc[:, 1])
    QoS = np.round(np.array(train.iloc[:, 2]),3)
    zuob0 = np.array(train.iloc[:, 3])
    zuob1 = np.array(train.iloc[:, 4])
    zuob2 = np.array(train.iloc[:, 5])
    zuob3 = np.array(train.iloc[:, 6])
    record=np.zeros([len(userID),7])
    for i in range(len(userID)):
        record[i][0]=int(serviceID[i])
        record[i][1]=int(userID[i])
        record[i][2]=np.round(QoS[i]*1000,3)

    A=5
    #hidden state
    M=5
    #记录的长度
    N=len(record)
    #惩罚
    w=50
    #初始化
    The_e = np.zeros([M, 1]) + (1 / M)
    The_a = np.zeros([M, 1]) + (1 / M)
    T_ijk=np.zeros([len(record), M,M])
    t_ijk=np.zeros([len(record),M,M])
    The_ijk=np.zeros([len(record),M,M])
    np.random.seed(10)
    Ba = np.random.rand(M, 5825)
    Be = np.random.rand(M, 339)
    #E
    def t_ijk2():
        for i in range(N):
            for j in range(M):
                for k in range(M):
                        ba = (math.gamma(5) ** 3) / math.gamma(15)
                        result = Ba[j][int(record[i][0])] * Be[k][int(record[i][1])] * The_a[j] * The_e[k] * (1 / ba)

                        sum = 1
                        for l in range(M):
                            sum = sum * ((The_a[l] * The_e[l]) ** 4)
                        result = result * sum
                        t_ijk[i,j,k]=result
    def The_ijk2():
        for i in range(N):
            for j in range(M):
                for k in range(M):
                    if  (Be[:,j]>np.mean(Be[:],axis=1)*0.8).all()==True and record[i,2]<500:
                        # print(i)
                        nu=Ca[int(record[j][0])]*Ce[int(record[k][1])]
                    else:
                        nu = Ca[int(record[j][0])] * Ce[int(record[k][1])]*w
                    # print((1/nu))
                    result=(1/nu)*np.exp(-(1/nu)*record[i][2])
                    The_ijk[i,j,k]=result

    def run_a():
        # print(The_ijk)
        for i in range(N):
            for j in range(M):
                for k in range(M):
                    T_ijk[i,j,k]=t_ijk[i,j,k]*The_ijk[i,j,k]
        # print(The_ijk)
        for i in range(N):
            T_ijk[i]=np.round(T_ijk[i]/T_ijk[i].sum(),4)
        return T_ijk

    def Mstep(T_ijk):
        for i in range(M):
            The_a[i]=((4*N)+T_ijk[:,i,:].sum())/((4*M+1)*N)
            The_e[i] = ((4 * N) + T_ijk[:, :, i].sum()) / ((4 * M + 1) * N)
        for i in range(M):
            for q in range(5825):
                       Ba[i][q]= T_ijk[list(np.where(record[:,0]==q)),i,:].sum()
        for i in range(M):
            for q in range(339):
                      Be[i][q] = T_ijk[list(np.where(record[:,1]==q)),:,i].sum()

        for i in range(M):
            Ba[i]=np.round((Ba[i]/sum(Ba[i]))*5825,4)
            Be[i] =np.round((Be[i]/sum(Be[i]))*339,4)



    def GD(T_ijk):
        global w
        # for q in range(5825):
        for i in range(N):
            for j in range(M):
                for k in range(M):
                        # if int(record[i][0])==q:
                        if j==k:
                            mide_result=T_ijk[i,j,k]*(record[i,2]/(Ce[int(record[i,1])]*(Ca[int(record[i,0])]**2))-(1/Ca[int(record[i,0])]))
                        else:
                            mide_result =T_ijk[i,j,k] * ((record[i,2] / (Ce[int(record[i,1])] * (Ca[int(record[i,0])] ** 2)))*(1/w)-(1/Ca[int(record[i,0])]))

                        Ca[int(record[i][0])]=Ca[int(record[i][0])]+mide_result*0.01
        # for q in range(339):
        for i in range(N):
            for j in range(M):
                for k in range(M):
                        # if int(record[i][1])==q:
                        if j==k:
                            mide_result2=T_ijk[i][j][k]*(record[i][2]/(Ca[int(record[i][0])]*(Ce[int(record[i][1])]**2))-(1/Ce[int(record[i][1])]))
                        else:
                            mide_result2 = T_ijk[i][j][k] * (
                                        (record[i][2] / (Ca[int(record[i][0])] * (Ce[int(record[i][1])] ** 2))) * (1 / w)-(1/Ce[int(record[i][1])]))
                        Ce[int(record[i][1])]=Ce[int(record[i][1])]+mide_result2*0.01
        mide=0
        for i in range(N):
            for j in range(M):
                for k in range(M):
                    if j!=k:
                        mide=mide+T_ijk[i,j,k]*((-1/w)+(record[i][2]/(Ca[int(record[i][0])]*Ce[int(record[i][1])]*w)))
        w=w+0.01*mide

    def L():
        zl=t_ijk.sum()*The_ijk.sum()*10
        return zl

    l=1
    l2=25
    bun=0
    while (bun<15):
        starttime = datetime.datetime.now()
        bun=bun+1
        t_ijk2()
        The_ijk2()
        re=run_a()
        Mstep(re)
        GD(re)
        s = '完成第 ' + str(bun) + '次'
        print(s)
        print(abs(l2-l)/l2)
        l = l2
        l2 = L()
        print("xcxxxxxxwwwwww")
        print(w)
        endtime = datetime.datetime.now()
        print((endtime - starttime).total_seconds())


    with open("LDA5_rt_train_%.2f.txt"% (dep), 'a')as F:
        print("train")
        for i in range(0, len(record)):
            x=int(record[i][0])
            u=int(record[i][1])
            F.writelines(str(int(record[i][0])) + "\t")
            F.writelines(str(int(record[i][1])) + "\t")
            F.writelines(str(QoS[i]) + "\t")
            F.writelines(str(zuob0[i]) + "\t")
            F.writelines(str(zuob1[i]) + "\t")
            F.writelines(str(zuob2[i]) + "\t")
            F.writelines(str(zuob3[i]) + "\t")
            F.writelines(str(Ba[0][x]) + "\t")
            F.writelines(str(Ba[1][x]) + "\t")
            F.writelines(str(Ba[2][x]) + "\t")
            F.writelines(str(Ba[3][x]) + "\t")
            F.writelines(str(Ba[4][x]) + "\t")
            # F.writelines(str(Ca[u][0]) + "\t")
            # F.writelines(str(Ba[5][x]) + "\t")
            # F.writelines(str(Ba[6][x]) + "\t")
            # F.writelines(str(Ba[7][x]) + "\t")
            # F.writelines(str(Ba[8][x]) + "\t")
            # F.writelines(str(Ba[9][x]) + "\t")

            # F.writelines(str(Ca[x]) + "\t")
            # F.writelines(str(Ba[5][x]) + "\t")
            # F.writelines(str(Ba[6][x]) + "\t")

            F.writelines(str(Be[0][u]) + "\t")
            F.writelines(str(Be[1][u]) + "\t")
            F.writelines(str(Be[2][u]) + "\t")
            F.writelines(str(Be[3][u]) + "\t")
            F.writelines(str(Be[4][u]) + "\n")
            # F.writelines(str(Ce[u][0]) + "\n")
            # F.writelines(str(Be[5][u]) + "\t")
            # F.writelines(str(Be[6][u]) + "\t")
            # F.writelines(str(Be[7][u]) + "\t")
            # F.writelines(str(Be[8][u]) + "\t")
            # F.writelines(str(Be[9][u]) + "\n")

         # F.writelines(str(Be[5][u]) + "\t")
            # F.writelines(str(Be[4][u]) + "\t")
            # F.writelines(str(Ce[u]) + "\n")

    train2 = pd.read_csv('0.02_d_rt_train.txt', sep='\t',float_precision='3')
    userID2 = np.array(train2.iloc[:, 0])
    serviceID2 = np.array(train2.iloc[:, 1])
    QoS2 = np.round(np.array(train2.iloc[:, 2]),3)
    record2=np.zeros([len(userID2),3])
    zuob02 = np.array(train2.iloc[:, 3])
    zuob12 = np.array(train2.iloc[:, 4])
    zuob22 = np.array(train2.iloc[:, 5])
    zuob32 = np.array(train2.iloc[:, 6])
    for i in range(len(userID2)):
        record2[i][0]=int(serviceID2[i])
        record2[i][1]=int(userID2[i])
        record2[i][2]=QoS2[i]

    with open("LDA5k_rt_test_%.2f.txt"% (dep), 'a')as tF:
        print("text")
        for i in range(0, len(record2)):
            x = int(record2[i][0])
            u = int(record2[i][1])
            tF.writelines(str(int(record2[i][0])) + "\t")
            tF.writelines(str(int(record2[i][1])) + "\t")
            tF.writelines(str(QoS2[i]) + "\t")
            tF.writelines(str(zuob02[i]) + "\t")
            tF.writelines(str(zuob12[i]) + "\t")
            tF.writelines(str(zuob22[i]) + "\t")
            tF.writelines(str(zuob32[i]) + "\t")
            tF.writelines(str(Ba[0][x]) + "\t")
            tF.writelines(str(Ba[1][x]) + "\t")
            tF.writelines(str(Ba[2][x]) + "\t")
            tF.writelines(str(Ba[3][x]) + "\t")
            tF.writelines(str(Ba[4][x]) + "\t")
            # tF.writelines(str(Ca[x][0]) + "\t")
            # tF.writelines(str(Ba[5][x]) + "\t")
            # tF.writelines(str(Ba[6][x]) + "\t")
            # tF.writelines(str(Ba[7][x]) + "\t")
            # tF.writelines(str(Ba[8][x]) + "\t")
            # tF.writelines(str(Ba[9][x]) + "\t")


            # tF.writelines(str(Ca[x]) + "\t")
            # tF.writelines(str(Ba[6][x]) + "\t")
            tF.writelines(str(Be[0][u]) + "\t")
            tF.writelines(str(Be[1][u]) + "\t")
            tF.writelines(str(Be[2][u]) + "\t")
            tF.writelines(str(Be[3][u]) + "\t")
            tF.writelines(str(Be[4][u]) + "\n")
            # tF.writelines(str(Ce[u][0]) + "\n")
            # tF.writelines(str(Be[5][u]) + "\t")
            # tF.writelines(str(Be[6][u]) + "\t")
            # tF.writelines(str(Be[7][u]) + "\t")
            # tF.writelines(str(Be[8][u]) + "\t")
            # tF.writelines(str(Be[9][u]) + "\n")

            # tF.writelines(str(Be[5][u]) + "\t")
            # tF.writelines(str(Be[4][u]) + "\t")
            # tF.writelines(str(Ce[u]) + "\n")


