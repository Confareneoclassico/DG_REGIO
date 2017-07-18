import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import random
from scipy import stats

df2 = pd.read_excel('Index slope 00-06.xlsx', sheetname='ERDF00-06', parse_cols = 'DW:EM')
df2.columns = df2.iloc[4]
df3 = df2.ix[5:253]
df4 = df3.set_index('Row Labels')
mu = pd.DataFrame(index = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, \
2014, 2015], data = (df4.columns.values-1999)*(df4[2015]['AT11']-df4[2000]['AT11'])/(2006-1999))
mu = mu.rename(columns={0: 'mu1'})
df4b = pd.concat([df4, mu.T])
for i in range (2007,2015):
    df4b[i]['mu1'] = 1
    
for i, row in df4b.iterrows():
    b = (df4b[2000]-df4b[2000]['mu1'], df4b[2001]-df4b[2001]['mu1'], df4b[2002]-df4b[2002]['mu1'],\
    df4b[2003]-df4b[2003]['mu1'], df4b[2004]-df4b[2004]['mu1'], df4b[2005]-df4b[2005]['mu1'], \
    df4b[2006]-df4b[2006]['mu1'], df4b[2007]-df4b[2007]['mu1'], df4b[2008]-df4b[2008]['mu1'], \
    df4b[2009]-df4b[2009]['mu1'], df4b[2010]-df4b[2010]['mu1'], df4b[2011]-df4b[2011]['mu1'],\
    df4b[2012]-df4b[2012]['mu1'], df4b[2013]-df4b[2013]['mu1'], df4b[2014]-df4b[2014]['mu1'],\
    df4b[2015]-df4b[2015]['mu1'])
    c = sum(b)
    df4b['muavg']= c
    
# Compute the relative scale
df4b['murel']=(max(df4b['muavg'])-df4b['muavg'])/(max(df4b['muavg'])-min(df4b['muavg']))
df6 = df4b.sort_values(['murel'], ascending = True)

# Prepare for the expenditure allocations
df21 = pd.read_excel('Index slope 00-06.xlsx', sheetname='ERDF00-06', parse_cols = 'J:Z')
df21.columns = df21.iloc[4]
df31 = df21.ix[5:253]
df41 = df31.set_index('Row Labels')
df51 = df41.T.fillna(0)

# Taken ITF6 as example
muA = df6['murel']['ITF6']

# Define the FiMin and FiMax distributions
FiMin = cp.Uniform(0.20, 0.40)
distributionFiMin = FiMin.sample(10000, rule="S")
np.random.shuffle(distributionFiMin)
FiMax = cp.Uniform(0.80, 1.00)
distributionFiMax = FiMax.sample(10000, rule="S")
np.random.shuffle(distributionFiMax)
distributionFi = distributionFiMax - muA * (distributionFiMax - distributionFiMin)

# Define the FiMinB and FiMaxB distributions
FiMinB = cp.Uniform(0.20, 0.40)
distributionFiMinB = FiMinB.sample(10000, rule="S")
np.random.shuffle(distributionFiMinB)
FiMaxB = cp.Uniform(0.80, 1.00)
distributionFiMaxB = FiMaxB.sample(10000, rule="S")
np.random.shuffle(distributionFiMaxB)
distributionFiAB2 = distributionFiMaxB - muA * (distributionFiMaxB - distributionFiMin)
distributionFiAB1 = distributionFiMax - muA * (distributionFiMax - distributionFiMinB)

# Define the residues A
B9 = []
for n in range(1,10):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B9.append(A(n))
C9 = np.array(B9)
D9 = np.insert(C9[0], [0, 0, 0, 0, 0, 0, 0, 0], 0)
E9 = np.insert(C9[1], [0, 0, 0, 0, 0, 0, 0], 0)
F9 = np.insert(C9[2], [0, 0, 0, 0, 0, 0], 0)
G9 = np.insert(C9[3], [0, 0, 0, 0, 0], 0)
H9 = np.insert(C9[4], [0, 0, 0, 0], 0)
I9 = np.insert(C9[5], [0, 0, 0], 0)
J9 = np.insert(C9[6], [0, 0], 0)
K9 = np.insert(C9[7], [0], 0)
L9 = np.array(C9[8])
M9 = (D9, E9, F9, G9, H9, I9, J9, K9, L9)

B8 = []
for n in range(1,9):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B8.append(A(n))
C8 = np.array(B8)
D8 = np.insert(C8[0], [0, 0, 0, 0, 0, 0, 0], 0)
E8 = np.insert(C8[1], [0, 0, 0, 0, 0, 0], 0)
F8 = np.insert(C8[2], [0, 0, 0, 0, 0], 0)
G8 = np.insert(C8[3], [0, 0, 0, 0], 0)
H8 = np.insert(C8[4], [0, 0, 0], 0)
I8 = np.insert(C8[5], [0, 0], 0)
J8 = np.insert(C8[6], [0], 0)
K8 = np.array(C8[7])
L8 = (D8, E8, F8, G8, H8, I8, J8, K8)

B7 = []
for n in range(1,8):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B7.append(A(n))
C7 = np.array(B7)
D7 = np.insert(C7[0], [0, 0, 0, 0, 0, 0], 0)
E7 = np.insert(C7[1], [0, 0, 0, 0, 0], 0)
F7 = np.insert(C7[2], [0, 0, 0, 0], 0)
G7 = np.insert(C7[3], [0, 0, 0], 0)
H7 = np.insert(C7[4], [0, 0], 0)
I7 = np.insert(C7[5], [0], 0)
J7 = np.array(C7[6])
K7 = (D7, E7, F7, G7, H7, I7, J7)
Q7 = random.choice(K7)

B6 = []
for n in range(1,7):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B6.append(A(n))
C6 = np.array(B6)
D6 = np.insert(C6[0], [0, 0, 0, 0, 0], 0)
E6 = np.insert(C6[1], [0, 0, 0, 0], 0)
F6 = np.insert(C6[2], [0, 0, 0], 0)
G6 = np.insert(C6[3], [0, 0], 0)
H6 = np.insert(C6[4], [0], 0)
I6 = np.array(C6[5])
J6 = (D6, E6, F6, G6, H6, I6)

B5 = []
for n in range(1,6):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B5.append(A(n))
C5 = np.array(B5)
D5 = np.insert(C5[0], [0, 0, 0, 0], 0)
E5 = np.insert(C5[1], [0, 0, 0], 0)
F5 = np.insert(C5[2], [0, 0], 0)
G5 = np.insert(C5[3], [0], 0)
H5 = np.array(C5[4])
I5 = (D5, E5, F5, G5, H5)

B4 = []
for n in range(1,5):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B4.append(A(n))
C4 = np.array(B4)
D4 = np.insert(C4[0], [0, 0, 0], 0)
E4 = np.insert(C4[1], [0, 0], 0)
F4 = np.insert(C4[2], [0], 0)
G4 = np.array(C4[3])
H4 = (D4, E4, F4, G4)

B3 = []
for n in range(1,4):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B3.append(A(n))
C3 = np.array(B3)
D3 = np.insert(C3[0], [0, 0], 0)
E3 = np.insert(C3[1], [0], 0)
F3 = np.array(C3[2])
G3 = (D3, E3, F3)

B2 = []
for n in range(1,3):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B2.append(A(n))
C2 = np.array(B2)
D2 = np.insert(C2[0], [0], 0)
E2 = np.array(C2[1])
F2 = (D2, E2)

B1 = []
for n in range(1,2):
    def A(n):
        return [(2**(j))/(2**n-1) for j in range(n)]
    B1.append(A(n))
C1 = np.array(B1)
E1 = (C1)

#Compile the expenditure adjustment database
R = pd.DataFrame(distributionFi, columns = ['Fi'])
df51['ITF6'][2009]
R[2009] = R['Fi']* (df51['ITF6'][2009]+df51['ITF6'][2010]+df51['ITF6'][2011]+df51['ITF6'][2012]+\
df51['ITF6'][2013]+df51['ITF6'][2014]+df51['ITF6'][2015])
R[2008] = R['Fi']* df51['ITF6'][2008] 
R[2007] = R['Fi']* df51['ITF6'][2007]
R[2006] = R['Fi']* df51['ITF6'][2006]
R[2005] = R['Fi']* df51['ITF6'][2005]
R[2004] = R['Fi']* df51['ITF6'][2004]
R[2003] = R['Fi']* df51['ITF6'][2003]
R[2002] = R['Fi']* df51['ITF6'][2002]
R[2001] = R['Fi']* df51['ITF6'][2001]
R[2000] = df51['ITF6'][2000]
R1 = R.copy()
R1[2009] = R[2009]

RAB2 = pd.DataFrame(distributionFiAB2, columns = ['Fi'])
RAB2[2009] = RAB2['Fi']* (df51['ITF6'][2009]+df51['ITF6'][2010]+df51['ITF6'][2011]+df51['ITF6'][2012]+\
df51['ITF6'][2013]+df51['ITF6'][2014]+df51['ITF6'][2015])
RAB2[2008] = RAB2['Fi']* df51['ITF6'][2008] 
RAB2[2007] = RAB2['Fi']* df51['ITF6'][2007]
RAB2[2006] = RAB2['Fi']* df51['ITF6'][2006]
RAB2[2005] = RAB2['Fi']* df51['ITF6'][2005]
RAB2[2004] = RAB2['Fi']* df51['ITF6'][2004]
RAB2[2003] = RAB2['Fi']* df51['ITF6'][2003]
RAB2[2002] = RAB2['Fi']* df51['ITF6'][2002]
RAB2[2001] = RAB2['Fi']* df51['ITF6'][2001]
RAB2[2000] = df51['ITF6'][2000]
R1AB2 = RAB2.copy()
R1AB2[2009] = RAB2[2009]

RAB1 = pd.DataFrame(distributionFiAB1, columns = ['Fi'])
RAB1[2009] = RAB1['Fi']* (df51['ITF6'][2009]+df51['ITF6'][2010]+df51['ITF6'][2011]+df51['ITF6'][2012]+\
df51['ITF6'][2013]+df51['ITF6'][2014]+df51['ITF6'][2015])
RAB1[2008] = RAB1['Fi']* df51['ITF6'][2008] 
RAB1[2007] = RAB1['Fi']* df51['ITF6'][2007]
RAB1[2006] = RAB1['Fi']* df51['ITF6'][2006]
RAB1[2005] = RAB1['Fi']* df51['ITF6'][2005]
RAB1[2004] = RAB1['Fi']* df51['ITF6'][2004]
RAB1[2003] = RAB1['Fi']* df51['ITF6'][2003]
RAB1[2002] = RAB1['Fi']* df51['ITF6'][2002]
RAB1[2001] = RAB1['Fi']* df51['ITF6'][2001]
RAB1[2000] = df51['ITF6'][2000]
R1AB1 = RAB1.copy()
R1AB1[2009] = RAB1[2009]

for i, row in R1.iterrows():
    n = (int(muA*8)+1, int(muA*7)+1, int(muA*6)+1, int(muA*5)+1, int(muA*4)+1, int(muA*3)+1, int(muA*2)+1, int(muA*1)+1)
    Q9 = random.choice(M9[0:n[0]])
    Q8 = random.choice(L8[0:n[1]])
    Q7 = random.choice(K7[0:n[2]])
    Q6 = random.choice(J6[0:n[3]])
    Q5 = random.choice(I5[0:n[4]])
    Q4 = random.choice(H4[0:n[5]])
    Q3 = random.choice(G3[0:n[6]])
    Q2 = random.choice(F2[0:n[7]])
    R1[2008] = R[2008] + (1 - R1['Fi']) * df51['ITF6'][2009] *  Q9[8]
    R1[2007] = R[2007] + (1 - R1['Fi']) * (df51['ITF6'][2009] *  Q9[7]\
    + df51['ITF6'][2008] * Q8[7])
    R1[2006] = R[2006] + (1 - R1['Fi']) * (df51['ITF6'][2009] *  Q9[6]\
    + df51['ITF6'][2008] * Q8[6] + df51['ITF6'][2007] * Q7[6])
    R1[2005] = R[2005] + (1 - R1['Fi']) * (df51['ITF6'][2009] *  Q9[5]\
    + df51['ITF6'][2008] * Q8[5] + df51['ITF6'][2007] * Q7[5] \
    + df51['ITF6'][2006] * Q6[5])
    R1[2004] = R[2004] + (1 - R1['Fi']) * (df51['ITF6'][2009] *  Q9[4]\
    + df51['ITF6'][2008] * Q8[4] + df51['ITF6'][2007] * Q7[4] \
    + df51['ITF6'][2006] * Q6[4] + df51['ITF6'][2005] * Q5[4])
    R1[2003] = R[2003] + (1 - R1['Fi']) * (df51['ITF6'][2009] *  Q9[3]\
    + df51['ITF6'][2008] * Q8[3] + df51['ITF6'][2007] * Q7[3] \
    + df51['ITF6'][2006] * Q6[3] + df51['ITF6'][2005] * Q5[3] \
    + df51['ITF6'][2004] * Q4[3])
    R1[2002] = R[2002] + (1 - R1['Fi']) * (df51['ITF6'][2009] *  Q9[2]\
    + df51['ITF6'][2008] * Q8[2] + df51['ITF6'][2007] * Q7[2] \
    + df51['ITF6'][2006] * Q6[2] + df51['ITF6'][2005] * Q5[2] \
    + df51['ITF6'][2004] * Q4[2] + df51['ITF6'][2003] * Q3[2])
    R1[2001] = R[2001] + (1 - R1['Fi']) * (df51['ITF6'][2009] *  Q9[1]\
    + df51['ITF6'][2008] * Q8[1] + df51['ITF6'][2007] * Q7[1] \
    + df51['ITF6'][2006] * Q6[1] + df51['ITF6'][2005] * Q5[1] \
    + df51['ITF6'][2004] * Q4[1] + df51['ITF6'][2003] * Q3[1] \
    + df51['ITF6'][2002] * Q2[1])
    R1[2000] = R[2000] + (1 - R1['Fi']) * (df51['ITF6'][2009] *  Q9[0]\
    + df51['ITF6'][2008] * Q8[0] + df51['ITF6'][2007] * Q7[0] \
    + df51['ITF6'][2006] * Q6[0] + df51['ITF6'][2005] * Q5[0] \
    + df51['ITF6'][2004] * Q4[0] + df51['ITF6'][2003] * Q3[0] \
    + df51['ITF6'][2002] * Q2[0] + df51['ITF6'][2001])

R1BA1 = RAB2.copy()
R1BA1[2009] = RAB2[2009]
for i4, row in R1BA1.iterrows():
    nb = (int(muA*8)+1, int(muA*7)+1, int(muA*6)+1, int(muA*5)+1, int(muA*4)+1, int(muA*3)+1, int(muA*2)+1, int(muA*1)+1)
    Q9b = random.choice(M9[0:nb[0]])
    Q8b = random.choice(L8[0:nb[1]])
    Q7b = random.choice(K7[0:nb[2]])
    Q6b = random.choice(J6[0:nb[3]])
    Q5b = random.choice(I5[0:nb[4]])
    Q4b = random.choice(H4[0:nb[5]])
    Q3b = random.choice(G3[0:nb[6]])
    Q2b = random.choice(F2[0:nb[7]])

    R1BA1[2008] = RAB2[2008] + (1 - R1BA1['Fi']) * df51['ITF6'][2009] *  Q9b[8]
    R1BA1[2007] = RAB2[2007] + (1 - R1BA1['Fi']) * (df51['ITF6'][2009] *  Q9b[7]\
    + df51['ITF6'][2008] * Q8b[7])
    R1BA1[2006] = RAB2[2006] + (1 - R1BA1['Fi']) * (df51['ITF6'][2009] *  Q9b[6]\
    + df51['ITF6'][2008] * Q8b[6] + df51['ITF6'][2007] * Q7b[6])
    R1BA1[2005] = RAB2[2005] + (1 - R1BA1['Fi']) * (df51['ITF6'][2009] *  Q9b[5]\
    + df51['ITF6'][2008] * Q8b[5] + df51['ITF6'][2007] * Q7b[5] \
    + df51['ITF6'][2006] * Q6b[5])
    R1BA1[2004] = RAB2[2004] + (1 - R1BA1['Fi']) * (df51['ITF6'][2009] *  Q9b[4]\
    + df51['ITF6'][2008] * Q8b[4] + df51['ITF6'][2007] * Q7b[4] \
    + df51['ITF6'][2006] * Q6b[4] + df51['ITF6'][2005] * Q5b[4])
    R1BA1[2003] = RAB2[2003] + (1 - R1BA1['Fi']) * (df51['ITF6'][2009] *  Q9b[3]\
    + df51['ITF6'][2008] * Q8b[3] + df51['ITF6'][2007] * Q7b[3] \
    + df51['ITF6'][2006] * Q6b[3] + df51['ITF6'][2005] * Q5b[3] \
    + df51['ITF6'][2004] * Q4b[3])
    R1BA1[2002] = RAB2[2002] + (1 - R1BA1['Fi']) * (df51['ITF6'][2009] *  Q9b[2]\
    + df51['ITF6'][2008] * Q8b[2] + df51['ITF6'][2007] * Q7b[2] \
    + df51['ITF6'][2006] * Q6b[2] + df51['ITF6'][2005] * Q5b[2] \
    + df51['ITF6'][2004] * Q4b[2] + df51['ITF6'][2003] * Q3b[2])
    R1BA1[2001] = RAB2[2001] + (1 - R1BA1['Fi']) * (df51['ITF6'][2009] *  Q9b[1]\
    + df51['ITF6'][2008] * Q8b[1] + df51['ITF6'][2007] * Q7b[1] \
    + df51['ITF6'][2006] * Q6b[1] + df51['ITF6'][2005] * Q5b[1] \
    +df51['ITF6'][2004] * Q4b[1] + df51['ITF6'][2003] * Q3b[1] \
    + df51['ITF6'][2002] * Q2b[1])
    R1BA1[2000] = RAB2[2000] + (1 - R1BA1['Fi']) * (df51['ITF6'][2009] *  Q9b[0]\
    + df51['ITF6'][2008] * Q8b[0] + df51['ITF6'][2007] * Q7b[0] \
    + df51['ITF6'][2006] * Q6b[0] + df51['ITF6'][2005] * Q5b[0] \
    + df51['ITF6'][2004] * Q4b[0] + df51['ITF6'][2003] * Q3b[0] \
    + df51['ITF6'][2002] * Q2b[0] + df51['ITF6'][2001])
    
R1BA2 = RAB1.copy()
R1BA2[2009] = RAB1[2009]
for i5, row in R1.iterrows():
    nc = (int(muA*8)+1, int(muA*7)+1, int(muA*6)+1, int(muA*5)+1, int(muA*4)+1, int(muA*3)+1, int(muA*2)+1, int(muA*1)+1)
    Q9c = random.choice(M9[0:nc[0]])
    Q8c = random.choice(L8[0:nc[1]])
    Q7c = random.choice(K7[0:nc[2]])
    Q6c = random.choice(J6[0:nc[3]])
    Q5c = random.choice(I5[0:nc[4]])
    Q4c = random.choice(H4[0:nc[5]])
    Q3c = random.choice(G3[0:nc[6]])
    Q2c = random.choice(F2[0:nc[7]])
    R1BA2[2008] = RAB1[2008] + (1 - R1BA2['Fi']) * df51['ITF6'][2009] *  Q9c[8]
    R1BA2[2007] = RAB1[2007] + (1 - R1BA2['Fi']) * (df51['ITF6'][2009] *  Q9c[7]\
    + df51['ITF6'][2008] * Q8c[7])
    R1BA2[2006] = RAB1[2006] + (1 - R1BA2['Fi']) * (df51['ITF6'][2009] *  Q9c[6]\
    + df51['ITF6'][2008] * Q8c[6] + df51['ITF6'][2007] * Q7c[6])
    R1BA2[2005] = RAB1[2005] + (1 - R1BA2['Fi']) * (df51['ITF6'][2009] *  Q9c[5]\
    + df51['ITF6'][2008] * Q8c[5] + df51['ITF6'][2007] * Q7c[5] \
    + df51['ITF6'][2006] * Q6c[5])
    R1BA2[2004] = RAB1[2004] + (1 - R1BA2['Fi']) * (df51['ITF6'][2009] *  Q9c[4]\
    + df51['ITF6'][2008] * Q8c[4] + df51['ITF6'][2007] * Q7c[4] \
    + df51['ITF6'][2006] * Q6c[4] + df51['ITF6'][2005] * Q5c[4])
    R1BA2[2003] = RAB1[2003] + (1 - R1BA2['Fi']) * (df51['ITF6'][2009] *  Q9c[3]\
    + df51['ITF6'][2008] * Q8c[3] + df51['ITF6'][2007] * Q7c[3] \
    + df51['ITF6'][2006] * Q6c[3] + df51['ITF6'][2005] * Q5c[3] \
    + df51['ITF6'][2004] * Q4c[3])
    R1BA2[2002] = RAB1[2002] + (1 - R1BA2['Fi']) * (df51['ITF6'][2009] *  Q9c[2]\
    + df51['ITF6'][2008] * Q8c[2] + df51['ITF6'][2007] * Q7c[2] \
    + df51['ITF6'][2006] * Q6c[2] + df51['ITF6'][2005] * Q5c[2] \
    + df51['ITF6'][2004] * Q4c[2] + df51['ITF6'][2003] * Q3c[2])
    R1BA2[2001] = RAB1[2001] + (1 - R1BA2['Fi']) * (df51['ITF6'][2009] *  Q9c[1]\
    + df51['ITF6'][2008] * Q8c[1] + df51['ITF6'][2007] * Q7c[1] \
    + df51['ITF6'][2006] * Q6c[1] + df51['ITF6'][2005] * Q5c[1] \
    +df51['ITF6'][2004] * Q4c[1] + df51['ITF6'][2003] * Q3c[1] \
    + df51['ITF6'][2002] * Q2c[1])
    R1BA2[2000] = RAB1[2000] + (1 - R1BA2['Fi']) * (df51['ITF6'][2009] *  Q9c[0]\
    + df51['ITF6'][2008] * Q8c[0] + df51['ITF6'][2007] * Q7c[0] \
    + df51['ITF6'][2006] * Q6c[0] + df51['ITF6'][2005] * Q5c[0] \
    + df51['ITF6'][2004] * Q4c[0] + df51['ITF6'][2003] * Q3c[0] \
    + df51['ITF6'][2002] * Q2c[0] + df51['ITF6'][2001])

# Sensitivity coefficient computation
varianceR1 = np.mean(R1**2)-np.mean(R1)**2
RSfimin = R1.multiply(R1BA1, axis = 0)
RSfimax = R1.multiply(R1BA2, axis = 0)
Sfimin = (RSfimin.sum(axis = 0)/10000 - np.mean(R1)**2)/varianceR1
Sfimax = (RSfimax.sum(axis = 0)/10000 - np.mean(R1)**2)/varianceR1
    
# Evaluate closure
Regression = sum(pd.DataFrame.mean(R1, axis = 0))
Closure = sum(df51['ITF6'])
Diff = (Closure - Regression)/Closure