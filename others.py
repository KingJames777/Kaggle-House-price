from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection  import  GridSearchCV,cross_val_score as cvs, KFold
from sklearn.preprocessing import RobustScaler
import pandas as pd;        import numpy as np;     import xgboost;     import lightgbm as lgb
from sklearn.linear_model import ElasticNet,Lasso,Ridge,BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor as rfr
from sklearn.feature_selection import SelectFromModel as sfm
from sklearn.kernel_ridge import KernelRidge
scorer='neg_mean_squared_error'

def rs(X,y=None):
    return RobustScaler().fit_transform(X,y)

def load_data():
    train=pd.read_csv('train_processed.csv');       test=pd.read_csv('test_processed.csv')
    return train,test

def CV(clf,X,y,seeds=range(3)):  ## shuffle data, and to alter method via seed.
    if type(seeds).__name__=='int':  ## shuffle only once.
        cv=np.sqrt(-cvs(clf,X,y,scoring=scorer,cv=KFold(10,True,seeds).split(X,y)))
        print('Mean:   ',round(np.mean(cv)*1e4,3),'\tMax:   ',round(np.max(cv)*1e4,3))
    else:
        median=[];      worst=[];       meen=[]
        for seed in seeds:
            cv=np.sqrt(-cvs(clf,X,y,scoring=scorer,cv=KFold(10,True,seed).split(X,y)))
            worst.append(np.max(cv));   meen.append(np.mean(cv))
        print('Mean3:   ',round(np.mean(meen)*1e4,3),'\tMax3:   ',round(np.mean(worst)*1e4,3))

def rmse(x):
    return np.sqrt(-x)

## missing data statistics.
def missing_stats(data):
    total = data.isnull().sum().sort_values(ascending=False)  ## total numbers of the missing.
    percent=(data.isnull().sum()/data.shape[0]).sort_values(ascending=False) ## percent
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def determine_n_estimators_for_lgb(X,y,params,n_estimators):
    dmat=lgb.Dataset(X,y)
    cvresult=lgb.cv(params,dmat,n_estimators,stratified=False,early_stopping_rounds=100,
                    metrics='rmse',verbose_eval=100,nfold=5)
    print(len(cvresult['rmse-mean']),'\n',1000*cvresult['rmse-mean'][-1])

def determine_n_estimators(X,y,clf):
    dmat=xgboost.DMatrix(X,y)
    cvresult = xgboost.cv(clf.get_xgb_params(), dmat, clf.get_xgb_params()['n_estimators'],5,
                       metrics='rmse',early_stopping_rounds=100)
    print(cvresult[['train-rmse-mean','test-rmse-mean','test-rmse-std']])

def paraTuning(clf,X,y):
    ## GBDT
##    params={'max_depth':[1,2,3,4,5]}
##    params={'subsample':[0.3,0.4,0.5],'min_samples_leaf':range(3,6)}
##    params={'max_features':['sqrt',0.3,0.4,0.8,0.2]}

    ## XGB
    ##gamma: threhold of gain,deciding whether to split any further.
    #min_child_weight: minimum weights of a leaf, similar to the minimum num of instances in a leaf
    ##reg_lambda: L2 norm for the weight of each instance
##    params={'reg_lambda':[1,1.5,2,2.5]}
##    params={'max_depth':range(3,5),'min_child_weight':range(2,5)}
##    params={'subsample':[x/100 for x in range(20,71,10)],
##            'colsample_bytree':[x/100 for x in range(10,20,3)]}

    ## LGB
    ##  subsample and subsample_freq should be set together,otherwise to no avail!!!
##    params={'subsample':[x/100.0 for x in range(5,101,5)],'subsample_freq':range(5,101,5),
##            'colsample_bytree':[x/100.0 for x in range(5,101,5)]}
##    params={'num_leaves':range(8,12,1),'max_depth':[3,4,5]}
##    params={'max_bin':range(130,180),'min_child_samples':[1,2,3,4,5]}
    params={'reg_lambda':[1,2,3],'reg_alpha':[1e-5,1e-7,0,0.1]}
##    params={'min_split_gain':[x/10 for x in range(0,11)],
##            'min_child_weight':[1e-8,1e-9,1e-7,1e-6,1e-5,1e-4,1e-3]}

    ## ElasticNet
##    params={'alpha': [x/10000 for x in range(1,10,2)],'l1_ratio': [x/100 for x in range(1,5,1)]}

    ## Lasso
##    params={'alpha': [x/100000 for x in range(1,3)]}

    ## Ridge
##    params={'alpha': range(0,50)}

    ## KernelRidge
##    params={'alpha':[x/1000 for x in range(5,10)]}
    
    ## the higher the score, the better the estimator!!!
    clfBest=GridSearchCV(clf,params,cv=7,scoring=scorer,n_jobs=4).fit(X,y)
    print(rmse(clfBest.best_score_),clfBest.best_params_)
    
def feature_importance(clf,columns,flag=1):
    if flag==1:
        df=pd.DataFrame({'features':columns,'importance':clf.feature_importances_}
                    ).sort_values('importance',ascending=False)
    else:
        df=pd.DataFrame({'features':columns,'importance':clf.coef_}
                    ).sort_values('importance',ascending=False)
        df=df.drop(df[abs(df.importance)<1e-6].index).reset_index(drop=True)
##    print(df[0:40]);    print(df[40:80]);       print(df[80:120]);     print(df[120:160]);      print(df[160:])
    return list(df.features.values)

def stacking(estimators,X1,X2,y,k=10):
    seed=0;     m,n=X1.shape;       t=X2.shape[0];     X_train,X_test=[],[]
    skf=KFold(k,random_state=seed)
    for clf in estimators:
        pred_train,pred_test=[],[]
        for train_index, test_index in skf.split(X1):
            clf.fit(X1[train_index],y[train_index])
            pred_train.extend(clf.predict(X1[test_index]))  ##  (m/5,) apply extend to join various arrays.
            pred_test.append(clf.predict(X2))  ##  (t,)  apply append for the convenience of calculating mean.
        X_train.append(pred_train)  ## (m,)
        X_test.append(np.sum(pred_test,axis=0)/k)  ## (t,)  ##sum and sum are different...
        print(1)
    return np.array(X_train).T,np.array(X_test).T

def newData(estimators,X,X_test,y):
    newX_train,newX_test=stacking(estimators,X,X_test,y)
    pd.DataFrame(newX_test).to_csv('newX_test212.csv',index=False)
    pd.DataFrame(newX_train).to_csv('newX_train212.csv',index=False)

def Select(clf,X,y,flag,seeds=range(3)):
    if flag==1:
        paraTuning(clf,X,y)
    elif flag==2:
        determine_n_estimators(X,y,clf)
    else:
        CV(clf,X,y,seeds)
    
def LGBT(dmat,params,flag):
    min_rmse=np.inf;    best_params = {};       rounds=0;       n_estimators=526
    if flag==1:
        for num_leaves in range(2,15):
            for max_depth in range(3,5):
                params['num_leaves'] = num_leaves
                params['max_depth'] = max_depth
                cvresult=lgb.cv(params,dmat,n_estimators,nfold=5,stratified=False,
                            early_stopping_rounds=100)
                mean_rmse=min(cvresult['rmse-mean'])
                if min_rmse>mean_rmse:
                    rounds=len(cvresult['rmse-mean'])
                    min_rmse=mean_rmse
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth
                    print(best_params,'\t',round(min_rmse,4))
        print('Final: ',rounds,'\t',best_params,'\t',round(min_rmse,4))
    elif flag==2:
        for max_bin in range(10,200,10):
            params['max_bin'] = max_bin
            for min_child_samples in range(2,6):
                params['min_child_samples'] = min_child_samples
                cvresult=lgb.cv(params,dmat,n_estimators,nfold=5,stratified=False,
                            early_stopping_rounds=100)
                mean_rmse=min(cvresult['rmse-mean'])
                if min_rmse>mean_rmse:
                    rounds=len(cvresult['rmse-mean'])
                    min_rmse=mean_rmse
                    best_params['max_bin'] = max_bin
                    best_params['min_child_samples'] = min_child_samples
                    print(best_params,'\t',round(min_rmse,4))
        print('Final: ',rounds,'\t',best_params,'\t',round(min_rmse,4))
    elif flag==3:
        for subsample in [x/100 for x in range(20,91,10)]:
            for subsample_freq in range(10,91,10):
                    for colsample_bytree in [x/100 for x in range(10,91,10)]:
                        params['subsample'] = subsample
                        params['subsample_freq'] = subsample_freq
                        params['colsample_bytree']=colsample_bytree
                        cvresult=lgb.cv(params,dmat,n_estimators,nfold=5,stratified=False,
                            early_stopping_rounds=100)
                        mean_rmse=min(cvresult['rmse-mean'])
                        if min_rmse>mean_rmse:
                            rounds=len(cvresult['rmse-mean'])
                            min_rmse=mean_rmse
                            best_params['subsample_freq'] = subsample_freq
                            best_params['subsample'] = subsample
                            best_params['colsample_bytree'] = colsample_bytree
                            print(best_params,'\t',round(min_rmse,4))
        print('Final: ',rounds,'\t',best_params,'\t',round(min_rmse,4))
    elif flag==4:
        for reg_alpha in [1e-7,1e-5,1e-3,0.1,0,1,2,3]:
            params['reg_alpha'] = reg_alpha
            for reg_lambda in range(1,6):
                params['reg_lambda'] = reg_lambda
                cvresult=lgb.cv(params,dmat,n_estimators,nfold=5,stratified=False,
                            early_stopping_rounds=100)
                mean_rmse=min(cvresult['rmse-mean'])
                if min_rmse>mean_rmse:
                    rounds=len(cvresult['rmse-mean'])
                    min_rmse=mean_rmse
                    best_params['reg_lambda'] = reg_lambda
                    best_params['reg_alpha'] = reg_alpha
                    print(best_params,'\t',round(min_rmse,4))
        print('Final: ',rounds,'\t',best_params,'\t',round(min_rmse,4))









