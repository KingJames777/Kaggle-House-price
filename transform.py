import numpy as np;         import pandas as pd
from scipy.stats import skew;   from scipy.special import boxcox1p

def delete(comb):
    comb=comb.drop(columns=['Utilities','Street','PoolQC','PoolArea','MoSold','Condition2',
                'Exterior1st','Exterior2nd','BsmtFinType2','MiscFeature','MasVnrArea','MiscVal'])
    return comb

def missingData(comb):
    for col in ['BsmtQual','BsmtExposure','BsmtCond','FireplaceQu','MasVnrType','Alley',
                'GarageFinish','GarageQual','GarageCond','GarageType','Fence','BsmtFinType1']: 
        comb[col].fillna('None',inplace=True)
        
    for col in ['KitchenQual','Electrical','MSZoning','Functional','SaleType']:
        comb[col].fillna(comb[col].mode()[0],inplace=True)
        
    for col in ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                'BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea']:
        comb[col].fillna(0,inplace=True)

    comb['LotFrontage']=comb.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    return comb 
 
def ordinalTransform(comb):
    for col in ['ExterQual','ExterCond','BsmtQual','BsmtCond','GarageQual','GarageCond',
                'HeatingQC','KitchenQual','FireplaceQu']:
        comb[col]=comb[col].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}).astype(np.int8)
    
    comb.LandSlope=comb.LandSlope.map({'Gtl':1,'Mod':0,'Sev':0})
    comb.CentralAir=comb.CentralAir.map({'Y':1,'N':0})
    comb.PavedDrive=comb.PavedDrive.map({'Y':5,'P':3,'N':1})
    comb.GarageFinish=comb.GarageFinish.map({'Fin':5,'RFn':3,'Unf':1,'None':0}) ##  NOT Rfn!!!
    comb.BsmtExposure=comb.BsmtExposure.map({'Gd':5,'Av':4,'Mn':3,'No':1,'None':0})
    comb.Functional=comb.Functional.map({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0})
    comb.Fence=comb.Fence.map({'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'None':0})
    
    return comb

def newFeatures(comb):
    comb['OverallQualCond']=comb['OverallQual']+comb['OverallCond']
    comb['TotalSF'] = comb['BsmtFinSF1']+comb['BsmtFinSF2'] + comb['1stFlrSF'] + comb['2ndFlrSF']
    comb['Baths']=comb['BsmtFullBath']+comb['FullBath']+(comb['HalfBath']+comb['BsmtHalfBath'])/2
    comb['Porch']=comb['OpenPorchSF']+comb['ScreenPorch']+comb['3SsnPorch']+comb.EnclosedPorch
    comb=comb.drop(columns=['OpenPorchSF','3SsnPorch','EnclosedPorch'])
    
    comb['LowQualFinSF']=comb['LowQualFinSF'].map(lambda x:0 if x==0 else 1)
    comb['EverRemod']=(comb.YearRemodAdd-comb.YearBuilt).map(lambda x:0 if x==0 else 1) # if ever remod
    comb['YearsAfterBuilt']=comb.YrSold-comb.YearBuilt+1  # Years since built
    comb['YearsAfterRemod']=(comb.YrSold-comb.YearRemodAdd).map(lambda x:0 if x<0 else x) # Years since remod
    
    comb['GarageSince']=(comb.YrSold-comb['GarageYrBlt']).map(lambda x:0 if x<0 else x)
    comb.loc[comb.GarageSince>2000,'GarageSince']=0
    comb=comb.drop(columns=['BsmtFinSF1','BsmtFinSF2','YearBuilt','YearRemodAdd','GarageYrBlt'])
    
    for col in ['LotFrontage','LotArea','BsmtFinSF','TotalBsmtSF','FullBath','GarageFinish',
                'GrLivArea','TotalSF','1stFlrSF','2ndFlrSF','Baths','TotRmsAbvGrd','Fireplaces','ScreenPorch',
                'GarageArea','GarageSince','OverallCond','FireplaceQu','BsmtExposure']: 
        comb[col+'Log']=np.log(1+comb[col])  ## log tranformation
        
    for  col in ['OverallQual','ExterQual','BsmtQual','GarageQual','KitchenQual','HeatingQC',
                 'GrLivAreaLog', 'TotalSFLog','1stFlrSFLog','YearsAfterBuilt']:
        comb['Square'+col]=comb[col]**2
    return comb

def Dummies(comb):
    comb.loc[comb['HouseStyle']=='2.5Unf','HouseStyle']='2.5Fin'
    comb.loc[comb['HouseStyle']=='1.5Fin','HouseStyle']='1.5Unf'

    comb.loc[comb['GarageType']=='2Types','GarageType']='CarPort'

    comb.loc[comb['LotConfig']=='FR3','LotConfig']='FR2'

    comb.loc[comb['Heating']!='GasA','Heating']='Oth'

    comb.loc[comb['Electrical']!='SBrkr','Electrical']='Oth'

    comb.loc[(comb['SaleType']!='WD')&(comb['SaleType']!='New'),'SaleType']='Oth'
    
    columns=['MSSubClass','MSZoning','Alley','LandContour','LotConfig','Neighborhood','BldgType',
             'BsmtFinType1','RoofStyle','HouseStyle','RoofMatl','Heating','YrSold','SaleType','Condition1',
             'SaleCondition','Foundation','MasVnrType','LotShape','Electrical','GarageType']
    for col in columns:
        comb=pd.concat([comb,pd.get_dummies(comb[col],prefix=col)],axis=1)
    comb=comb.drop(columns=columns)
    
    return comb

def unSkew(comb):
    target=comb.SalePrice;      comb=comb.drop(columns='SalePrice')
    numeric_feats = comb.dtypes[comb.dtypes != 'object'].index
    skewed_feats = comb[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_features = skewed_feats[abs(skewed_feats) > 0.5].index
    lam = 0.15
    for feat in skewed_features:
        comb[feat] = boxcox1p(comb[feat], lam)
    comb=pd.concat([comb,target],axis=1)
    return comb



