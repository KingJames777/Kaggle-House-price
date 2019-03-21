from others import *

train,test=load_data();         y=np.log(train['SalePrice'].values)
X=pd.read_csv('nX.csv').values;     X_test=pd.read_csv('nX_test.csv').values
dmat=lgb.Dataset(X,y)
params={
        'num_leaves':6,
        'max_depth':3,
        'col_sample_bytree':0.2,
        'subsample':0.8,
        'subsample_freq':10,
        'reg_lambda':1,
        'min_child_samples':3,
        'max_bin':10,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'nthread':3,
        'learning_rate':0.01,
        'seed':0}
determine_n_estimators_for_lgb(X,y,params,5000)
##LGBT(dmat,params,4)


##clf=lgb.LGBMRegressor(n_estimators=133,num_leaves=8,max_depth=3,
##            colsample_bytree=0.1,subsample=0.8,subsample_freq=10,max_bin=152,
##            reg_lambda=1,min_child_samples=2,learning_rate=0.1,random_state=0,
##                      objection='regression')
##Select(clf,X,y,3)
##Mean3:    0.10947999576 	    Max3:    0.1268057317














##clf1=xgboost.XGBRegressor(n_estimators=2800,learning_rate=0.01,max_depth=3,
##        subsample=0.4,col_sample_bytree=0.2,reg_lambda=2,
##                          min_child_weight=3,random_state=0)
##Mean3:    0.111539271        Max3:    0.129212393
##clf3=GradientBoostingRegressor(loss='huber',n_estimators=3200,learning_rate=0.02,
##        max_depth=3,max_features='sqrt',min_samples_leaf=3,subsample=0.4,
##                              random_state=0)
##Mean3:    0.107883285085 	    Max3:    0.1258323141
