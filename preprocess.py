from transform import *

if __name__=='__main__':
    train=pd.read_csv('train.csv');         test=pd.read_csv('test.csv')
    train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index).reset_index(drop=True)
    train=train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index).reset_index(drop=True)
    test['SalePrice']=0;    comb=pd.concat([train,test],ignore_index=True)

    comb=delete(comb)
    comb=missingData(comb)
    comb=unSkew(comb)
    comb=ordinalTransform(comb)
    comb=newFeatures(comb)
    comb=Dummies(comb)

    trainProces=comb[comb.SalePrice!=0].drop(columns='Id')
    trainProces.to_csv('train_processed.csv',index=False)
    testProces=comb[comb.SalePrice==0]
    testProces.drop(columns='SalePrice').to_csv('test_processed.csv',index=False)
