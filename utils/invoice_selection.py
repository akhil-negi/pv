import json
#import pickle
import numpy as np
import pandas as pd
from utils.helper import *
import os
from sklearn.externals import joblib

class fit_for_auction_check:

    	
    def __init__(self):
        self.mandate_cols=['Invoice Value','Invoice Load Date in PV','Original Invoice Pay Schedule Date']
        self.cat_dict=joblib.load('models/cat_dict.pk')
        self.mapper=joblib.load('models/mapper.pk')
        self.mapper_no_supp=joblib.load('models/mapper_no_supp.pk')
        self.rf=joblib.load('models/rf.pk')
        self.rf_no_supp=joblib.load('models/rf_no_supp.pk')
        
            
    def preprocess(self,df1):
        df=df1.copy(deep=True)        
        cat_dict=self.cat_dict
        if {'Supplier Category','SUPP_ID'}.issubset(set(df.columns)):
            mapper=self.mapper   
        else:
            mapper=self.mapper_no_supp
        if not set(self.mandate_cols).issubset(set(df.columns)):
            print(f"ERROR_MSG : Please provide {mandate_cols} in the input.")
            return
        df_proc=df.drop(['CUST_ID','Auction Created?','AUCTION_ID'],axis=1).copy(deep=True)
        df_discarded=df_proc.loc[df_proc[pd.Series(df_proc.isna().sum(axis=1)).astype('bool')].index]
        df_proc.dropna(inplace=True)
        df_proc["Invoice Load Date in PV"]=pd.to_datetime(df_proc["Invoice Load Date in PV"])
        df_proc["Original Invoice Pay Schedule Date"]=df_proc["Original Invoice Pay Schedule Date"].str.replace('/',"-")
        df_proc["Original Invoice Pay Schedule Date"]=pd.to_datetime(df_proc["Original Invoice Pay Schedule Date"],format='%d-%m-%Y',errors = 'coerce')
        df_proc['time_left']=(df_proc["Original Invoice Pay Schedule Date"]-df_proc["Invoice Load Date in PV"]).apply(lambda x:x.days)
        df_discarded=pd.concat([df_discarded,df_proc[(df_proc.time_left<=0).astype('bool')|(df_proc['Invoice Value']<0).astype('bool')]],axis=0)
        df_proc=df_proc[(df_proc.time_left>0).astype('bool')&(df_proc['Invoice Value']>0).astype('bool')]
        self.invoice_ids=df_proc.INV_ID
        df_proc.drop('INV_ID',axis=1,inplace=True)
        add_datepart(df_proc,'Invoice Load Date in PV')
        add_datepart(df_proc,'Original Invoice Pay Schedule Date')
        cat_vars=[col for col in df_proc.columns if col not in ['Invoice Value','time_left']]
        cont_vars=['Invoice Value','time_left']
        for col in cat_vars:
            df_proc[f'{col}']=df_proc[f'{col}'].astype('category',categories=
                                    (cat_dict[f'{col}']))
        for v in cont_vars: df_proc[v] = df_proc[v].astype('float32')
        df_proc,_,_,_  = proc_df(df_proc,do_scale=True,mapper=mapper)
        return(df_proc,df_discarded)
    
    def predict(self,df_proc):
        if {'Supplier Category','SUPP_ID'}.issubset(set(df_proc.columns)):
            loaded_model = self.rf

            preds=loaded_model.predict(df_proc)
            preds=np.array(preds, dtype=bool)
            probs=loaded_model.predict_proba(df_proc)[:,1]
            probs=pd.DataFrame(list(zip(self.invoice_ids,probs.tolist(),preds.tolist())),columns=['INV_ID','Participation_probability','Predicted_Participation'])
            return probs,loaded_model
        elif not {'Supplier Category','SUPP_ID'}.issubset(set(df_proc.columns)):            
            loaded_model = self.rf_no_supp
            probs=loaded_model.predict_proba(df_proc)[:,1]
            probs=pd.DataFrame(list(zip(self.invoice_ids,probs.tolist())),columns=['INV_ID','Participation_probability'])
            return (probs,loaded_model)
        else:
            print("ERROR_MSG : Please check the input columns")
            return

