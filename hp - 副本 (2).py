from others import *
train,test=load_data();         y=np.log(train['SalePrice'].values);        ID=test['Id'].values
train.drop(columns='SalePrice',inplace=True);       columns=train.columns 
test=test.drop('Id',axis=1);       X=train.values;             X_test=test.values

##clf=Lasso(0.0005,max_iter=200000).fit(X,y)
##col=feature_importance(clf,columns,0)
##X=train[col].values;        X_test=test[col].values
X=rs(X)
clf=Ridge(alpha=10,max_iter=20000)
CV(clf,X,y,3)

##clf=Ridge(alpha=1.5,max_iter=20000)      
##Mean3:    0.1074581154        Max3:       0.1230832545

##clf=Lasso(0.0001,max_iter=200000)     
##Mean3:    0.107699673         Max3:    0.12341570077

##clf=ElasticNet(0.0001,0.01,random_state=0,max_iter=200000)
##Mean3:    0.1074509003        Max3:    0.123113084475

##clf=BayesianRidge(tol=0.5,alpha_2=1.8,lambda_2=0.15)
##Mean3:    1075.2715908 	Max3:    1233.2129370

##clf=KernelRidge(0.0001)
##Mean3:    0.1079867557	Max3:    0.12374463866



