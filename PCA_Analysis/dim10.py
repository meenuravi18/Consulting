
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import matplotlib as plt
import math 
from sklearn import decomposition


data = pd.read_excel("chile6-10.xlsx",sheet_name=0)
criteria=[]
regions=[]
group1=[]

for cols in data.columns:
	criteria.append(cols)
i=46
while (i<51):
    group1.append(criteria[i])
    i+=1

for index,row in data.iterrows():
    regions.append(row['Regions'])
print(group1)
num=data.iloc[0:15,46:51]
num=num.to_numpy()

data1=pd.DataFrame(num,index=regions,columns=group1)
data1 = data1.replace('-','', regex = True)
data1 = data1.replace(',','', regex = True)


x_std=StandardScaler().fit_transform(data1)

features=x_std.T
covariance_matrix=np.cov(features)


eig_values,eig_vectors=np.linalg.eig(covariance_matrix)
idx = eig_values.argsort()[::-1]   
eig_values = eig_values[idx]
eig_vectors = eig_vectors[:,idx]
var_explained=[]
exp_var=[]
print(eig_values)
for i in eig_values:
    var_explained.append(i/sum(eig_values)*100)
    exp_var.append(i.real)
cumulative_variance_explained=np.cumsum(var_explained)
for i in range(0,len(eig_values)):
	print(eig_values[i],var_explained[i],cumulative_variance_explained[i])
exp_var=exp_var[0:2]


pca=decomposition.PCA(n_components=2)

X=pca.fit_transform(x_std)


component_matrix = pd.DataFrame(pca.components_.T, columns=['PC1','PC2'], index=group1)

coeffs = pca.components_.T / np.sqrt(exp_var)

coeffs_matrix = pd.DataFrame(coeffs, columns=['PC1','PC2'], index=group1)

print(component_matrix)
print(coeffs_matrix)

variance=[]
ctr=0
for i in var_explained:
	if ctr<2:
		variance.append(i.real)
	ctr+=1

ctr=1
values=[]
for i in range(0,len(num)):
	temp=[]
	for j in range(0,len(num[i])):
		v=int(num[i][j])
		temp.append(v)
	values.append(temp)


coeffs_Mat=coeffs.T


overallRankings=[]
b=0
while b<15:
	a=0
	rankings=[]
	while a<2:
		L_vals=[]
		sums=[]
		coeffs_=coeffs_Mat[a]
		for i in range(0,len(x_std[0])):
			L_vals.append(x_std[b][i]*coeffs_[i])
		sums.append(sum(L_vals)*(variance[a]/100))
		rankings.append(sum(sums))
		a+=1
	overallRankings.append(sum(rankings))
	b+=1
regionsR=[]
for i in range(0,len(overallRankings)):
	regionsR.append([regions[i],overallRankings[i]])
print(regionsR)
sortedRegions=sorted(regionsR, key=lambda x: x[1],reverse=True)

print (pd.DataFrame(sortedRegions))
pd.DataFrame(sortedRegions).to_clipboard(sep=',', index=False)  


