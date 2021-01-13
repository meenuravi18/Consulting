
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import matplotlib as plt
import math 
from sklearn import decomposition
from tabulate import tabulate
'''
Perform a Principal Component Analysis (PCA) for the dataset provided, and come up with:
A ranking for each one of the five dimensions considered, such that regions are ranked within each (five rankings in total).
'''
'''
METHODOLGY:
Import necessary libraries and modules
Read excel file using pandas
Get the criteria for the specific dimension
Get the regions
Create a new dataframe with the regions, czzariteria and data associated
Scale the data using sklearn’s standard scaler and fit the data
Get the covariance matrix using  numpy’s cov function
Calculate the eigenvalues and eigenvectors using np.linalg.eig of the covariance matrix
 
I decided to use the first X (X=5) components
I found the component matrix from eigenvectors and the first five components
Then I found the coefficients matrix using the component matrix divided by square root of eigenvalues

Then I found the L values using the dataset and multiplied by each element in coefficient matrix and summed this for each component
Then I took the 5 L values and multiplied each with the respective percentage of variation
Then I rank the regions according to respective L values. The higher the value the higher the rank

'''
data = pd.read_excel("chile_data.xlsx",sheet_name=0)
criteria=[]
regions=[]
group1=[]
for cols in data.columns:
    criteria.append(cols)
i=2
while (i<26):
    group1.append(criteria[i])
    i+=1

for index,row in data.iterrows():
    regions.append(row['VARIABLE'])

num=data.iloc[0:15,2:26]
num=num.to_numpy()

data1=pd.DataFrame(num,index=regions,columns=group1)

X=scale(data1)

x_std=StandardScaler().fit_transform(data1)

features=x_std.T
covariance_matrix=np.cov(features)


eig_values,eig_vectors=np.linalg.eig(covariance_matrix)

var_explained=[]
exp_var=[]
for i in eig_values:
    var_explained.append(i/sum(eig_values)*100)
    exp_var.append(i.real)
cumulative_variance_explained=np.cumsum(var_explained)
for i in range(0,len(eig_values)):
	print(eig_values[i],var_explained[i],cumulative_variance_explained[i])
exp_var=exp_var[0:5]


pca=decomposition.PCA(n_components=5)

X=pca.fit_transform(X)


component_matrix = pd.DataFrame(pca.components_.T, columns=['PC1','PC2','PC3','PC4',"PC5"], index=group1)

coeffs = pca.components_.T / np.sqrt(exp_var)

coeffs_matrix = pd.DataFrame(coeffs, columns=['PC1','PC2','PC3','PC4',"PC5"], index=group1)

print(component_matrix)
print(coeffs_matrix)

variance=[]
ctr=0
for i in var_explained:
	if ctr<5:
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
	while a<5:
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


