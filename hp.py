from others import *
train,test=load_data();         y=np.log(train['SalePrice'].values);        ID=test['Id'].values
X,X_test=pd.read_csv('Ax.csv'),pd.read_csv('Bx.csv')

clf1=KernelRidge(1)
clf2=Ridge(alpha=0.8)
clf3=BayesianRidge(tol=0.2,alpha_2=80,lambda_2=0.15)
clf4=ElasticNet(0.0003,0.07,max_iter=20000)
clf5=Lasso(0.0003,max_iter=200000)
clf6=GradientBoostingRegressor('huber',0.01,770,0.25,max_depth=2,
            max_features='sqrt',min_samples_split=5,random_state=0)
clf7=lgb.LGBMRegressor('gbdt',3,2,0.01,800,colsample_bytree=0.3,
            subsample=0.6,subsample_freq=10,reg_lambda=1,
            min_child_samples=4,random_state=2)
estimators=[clf1,clf2,clf3,clf4,clf5,clf6,clf7]

##Select(clf,X,y,3)
##clf1=KernelRidge(0.9)
##clf2=Ridge(alpha=0.9)
##clf3=BayesianRidge(tol=0.2,alpha_2=80,lambda_2=0.15)
##clf4=ElasticNet(0.0003,0.05,max_iter=20000)
##clf5=Lasso(0.0003,max_iter=200000)
##clf6=GradientBoostingRegressor('huber',0.01,770,0.25,max_depth=2,
##            max_features='sqrt',min_samples_split=5,random_state=0)
##clf7=lgb.LGBMRegressor('gbdt',3,2,0.01,800,colsample_bytree=0.3,
##            subsample=0.6,subsample_freq=10,reg_lambda=1,
##            min_child_samples=4,random_state=2)
##clf8=xgboost.XGBRegressor(2,0.01,850,subsample=0.37,
##    col_sample_bytree=0.5,colsample_bylevel=0.2,random_state=2)

##estimators=[clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8]
##
pred=[]
for clf in estimators:
    pred.append(np.exp(clf.fit(X,y).predict(X_test)))

pred=pred[0]*0.12+(pred[1]+pred[2]+pred[3]+pred[4])*0.12+(pred[5]+pred[6])*0.2
subm=pd.DataFrame({'ID':ID,'SalePrice':pred})
subm.to_csv('Pred.csv',index=False)







