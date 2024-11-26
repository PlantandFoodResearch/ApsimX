# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import APSIMGraphHelpers as AGH
import GraphHelpers as GH
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.dates as mdates
import MathsUtilities as MUte
import shlex # package to construct the git command to subprocess format
import subprocess 
#import ProcessWheatFiles as pwf
import xmltodict, json
import sqlite3
import scipy.optimize 
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt import load
from skopt.plots import plot_convergence
import matplotlib.gridspec as gridspec
import time

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
# %matplotlib inline
# -

BaseSimNames =['Roberts1988InitT1oCInitPp8hDurat0FinalT12oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp8hDurat10FinalT12oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp8hDurat30FinalT12oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp8hDurat60FinalT12oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp8hDurat0FinalT12oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp8hDurat10FinalT12oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp8hDurat30FinalT12oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp8hDurat60FinalT12oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp8hDurat0FinalT12oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp8hDurat10FinalT12oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp8hDurat30FinalT12oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp8hDurat60FinalT12oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp16hDurat0FinalT12oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp16hDurat10FinalT12oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp16hDurat30FinalT12oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp16hDurat60FinalT12oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp16hDurat0FinalT12oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp16hDurat10FinalT12oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp16hDurat30FinalT12oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp16hDurat60FinalT12oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp16hDurat0FinalT12oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp16hDurat10FinalT12oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp16hDurat30FinalT12oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp16hDurat60FinalT12oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp8hDurat0FinalT19oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp8hDurat10FinalT19oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp8hDurat30FinalT19oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp8hDurat60FinalT19oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp8hDurat0FinalT19oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp8hDurat10FinalT19oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp8hDurat30FinalT19oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp8hDurat60FinalT19oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp8hDurat0FinalT19oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp8hDurat10FinalT19oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp8hDurat30FinalT19oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp8hDurat60FinalT19oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp16hDurat0FinalT19oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp16hDurat10FinalT19oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp16hDurat30FinalT19oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp16hDurat60FinalT19oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp16hDurat0FinalT19oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp16hDurat10FinalT19oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp16hDurat30FinalT19oCFinalPp11hCv',
'Roberts1988InitT5oCInitPp16hDurat60FinalT19oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp16hDurat0FinalT19oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp16hDurat10FinalT19oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp16hDurat30FinalT19oCFinalPp11hCv',
'Roberts1988InitT9oCInitPp16hDurat60FinalT19oCFinalPp11hCv',
'Roberts1988InitT1oCInitPp8hDurat0FinalT12oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp8hDurat10FinalT12oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp8hDurat30FinalT12oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp8hDurat60FinalT12oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp8hDurat0FinalT12oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp8hDurat10FinalT12oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp8hDurat30FinalT12oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp8hDurat60FinalT12oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp8hDurat0FinalT12oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp8hDurat10FinalT12oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp8hDurat30FinalT12oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp8hDurat60FinalT12oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp16hDurat0FinalT12oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp16hDurat10FinalT12oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp16hDurat30FinalT12oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp16hDurat60FinalT12oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp16hDurat0FinalT12oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp16hDurat10FinalT12oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp16hDurat30FinalT12oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp16hDurat60FinalT12oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp16hDurat0FinalT12oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp16hDurat10FinalT12oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp16hDurat30FinalT12oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp16hDurat60FinalT12oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp8hDurat0FinalT19oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp8hDurat10FinalT19oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp8hDurat30FinalT19oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp8hDurat60FinalT19oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp8hDurat0FinalT19oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp8hDurat10FinalT19oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp8hDurat30FinalT19oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp8hDurat60FinalT19oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp8hDurat0FinalT19oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp8hDurat10FinalT19oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp8hDurat30FinalT19oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp8hDurat60FinalT19oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp16hDurat0FinalT19oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp16hDurat10FinalT19oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp16hDurat30FinalT19oCFinalPp16hCv',
'Roberts1988InitT1oCInitPp16hDurat60FinalT19oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp16hDurat0FinalT19oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp16hDurat10FinalT19oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp16hDurat30FinalT19oCFinalPp16hCv',
'Roberts1988InitT5oCInitPp16hDurat60FinalT19oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp16hDurat0FinalT19oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp16hDurat10FinalT19oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp16hDurat30FinalT19oCFinalPp16hCv',
'Roberts1988InitT9oCInitPp16hDurat60FinalT19oCFinalPp16hCv',
'Summerfield1985InitT1oCDurat0FinalTmax18oCFinalTmin5oCFinalPp10hCv',
'Summerfield1985InitT1oCDurat0FinalTmax28oCFinalTmin5oCFinalPp10hCv',
'Summerfield1985InitT1oCDurat0FinalTmax18oCFinalTmin13oCFinalPp10hCv',
'Summerfield1985InitT1oCDurat0FinalTmax28oCFinalTmin13oCFinalPp10hCv',
'Summerfield1985InitT1oCDurat0FinalTmax18oCFinalTmin5oCFinalPp13hCv',
'Summerfield1985InitT1oCDurat0FinalTmax18oCFinalTmin13oCFinalPp13hCv',
'Summerfield1985InitT1oCDurat0FinalTmax28oCFinalTmin5oCFinalPp13hCv',
'Summerfield1985InitT1oCDurat0FinalTmax28oCFinalTmin13oCFinalPp13hCv',
'Summerfield1985InitT1oCDurat0FinalTmax18oCFinalTmin5oCFinalPp16hCv',
'Summerfield1985InitT1oCDurat0FinalTmax18oCFinalTmin13oCFinalPp16hCv',
'Summerfield1985InitT1oCDurat0FinalTmax28oCFinalTmin5oCFinalPp16hCv',
'Summerfield1985InitT1oCDurat0FinalTmax28oCFinalTmin13oCFinalPp16hCv',
'Summerfield1985InitT1oCDurat30FinalTmax18oCFinalTmin5oCFinalPp10hCv',
'Summerfield1985InitT1oCDurat30FinalTmax28oCFinalTmin5oCFinalPp10hCv',
'Summerfield1985InitT1oCDurat30FinalTmax18oCFinalTmin13oCFinalPp10hCv',
'Summerfield1985InitT1oCDurat30FinalTmax28oCFinalTmin13oCFinalPp10hCv',
'Summerfield1985InitT1oCDurat30FinalTmax18oCFinalTmin5oCFinalPp13hCv',
'Summerfield1985InitT1oCDurat30FinalTmax18oCFinalTmin13oCFinalPp13hCv',
'Summerfield1985InitT1oCDurat30FinalTmax28oCFinalTmin5oCFinalPp13hCv',
'Summerfield1985InitT1oCDurat30FinalTmax28oCFinalTmin13oCFinalPp13hCv',
'Summerfield1985InitT1oCDurat30FinalTmax18oCFinalTmin5oCFinalPp16hCv',
'Summerfield1985InitT1oCDurat30FinalTmax18oCFinalTmin13oCFinalPp16hCv',
'Summerfield1985InitT1oCDurat30FinalTmax28oCFinalTmin5oCFinalPp16hCv',
'Summerfield1985InitT1oCDurat30FinalTmax28oCFinalTmin13oCFinalPp16hCv',
'Roberts1986Pp10to16hDurat41Cv',
'Roberts1986Pp10to16hDurat37Cv',
'Roberts1986Pp10to16hDurat33Cv',
'Roberts1986Pp10to16hDurat29Cv',
'Roberts1986Pp10to16hDurat25Cv',
'Roberts1986Pp10to16hDurat21Cv',
'Roberts1986Pp10to16hDurat17Cv',
'Roberts1986Pp10to16hDurat13Cv',
'Roberts1986Pp10to16hDurat9Cv',
'Roberts1986Pp10to16hDurat0Cv',
'Roberts1986Pp16to10hDurat45Cv',
'Roberts1986Pp16to10hDurat41Cv',
'Roberts1986Pp16to10hDurat37Cv',
'Roberts1986Pp16to10hDurat33Cv',
'Roberts1986Pp16to10hDurat29Cv',
'Roberts1986Pp16to10hDurat25Cv',
'Roberts1986Pp16to10hDurat21Cv',
'Roberts1986Pp16to10hDurat17Cv',
'Roberts1986Pp16to10hDurat13Cv',
'Roberts1986Pp16to10hDurat9Cv',
'Roberts1986Pp16to10hDurat0Cv']

# +
BaseLentilFilePath = r"C:/GitHubRepos/ApsimX/Prototypes/Lentil/Lentil.apsimx"
WorkingLentilFilePath = r"C:/GitHubRepos/ApsimX/Prototypes/Lentil/LentilOptItter.apsimx"
DBFilePath = r"C:/GitHubRepos/ApsimX/Prototypes/Lentil/LentilOptItter.db"

paramNames = ['[Phenology].JuvenileBase.FixedValue', 
              '[Phenology].VernSensitivity.FixedValue', 
              '[Phenology].InductiveBase.FixedValue', 
              '[Phenology].InductivePpSensitivity.FixedValue',
              '[Phenology].EarlyReproductiveBase.FixedValue', 
              '[Phenology].EarlyReproductivePpSensitivity.FixedValue']
              # '[Phenology].ThermalTime.PostEmergence.MultiplyFunction.WangEagleTempScale.Response.MinTemp',
              # '[Phenology].ThermalTime.PostEmergence.MultiplyFunction.WangEagleTempScale.Response.OptTemp',
              # '[Phenology].ThermalTime.PostEmergence.MultiplyFunction.WangEagleTempScale.Response.MaxTemp',
              # '[Phenology].AccumulatedVernalisation.DailyVernalisation.Response.MinTemp',
              # '[Phenology].AccumulatedVernalisation.DailyVernalisation.Response.OptTemp',
              # '[Phenology].AccumulatedVernalisation.DailyVernalisation.Response.MaxTemp']

FittingVariables = ['Lentil.Phenology.StartBuddingDAS','Lentil.Phenology.StartFloweringDAS']        

BlankManager = {'$type': 'Models.Manager, Models',
            'CodeArray': [],
            'Parameters': [],
            'Name': 'SetCropParameters',
            'Enabled': True,
            'ReadOnly': False}

SetCropParams = {
          "$type": "Models.Manager, Models",
          "CodeArray": [],
          "Parameters": [],
          "Name": "SetCropParameters",
          "Enabled": True,
          "ReadOnly": False}

def AppendModeltoModelofTypeAndDeleteOldIfPresent(Parent,TypeToAppendTo,ModelToAppend):
    try:
        for child in Parent['Children']:
            if child['$type'] == TypeToAppendTo:
                pos = 0
                for g in child['Children']:
                    if g['Name'] == ModelToAppend['Name']:
                        del child['Children'][pos]
                        #print('Model ' + ModelToAppend['Name'] + ' found and deleted')
                    pos+=1
                child['Children'].append(ModelToAppend)
                return True
            else:
                Parent = AppendModeltoModelofTypeAndDeleteOldIfPresent(child,TypeToAppendTo,ModelToAppend)
        return Parent
    except:
        return Parent
    
def AppendModeltoModelofType(Parent,TypeToAppendTo,NameToAppendTo,ModelToAppend):
    try:
        for child in Parent['Children']:
            if (child['$type'] == TypeToAppendTo) and (child['Name'] == NameToAppendTo):
                child['Children'].append(ModelToAppend)
                return True
            else:
                Parent = AppendModeltoModelofType(child,TypeToAppendTo,NameToAppendTo,ModelToAppend)
        return Parent
    except:
        return Parent
    
def findNextChild(Parent,ChildName):
    if len(Parent['Children']) >0:
        for child in range(len(Parent['Children'])):
            if Parent['Children'][child]['Name'] == ChildName:
                return Parent['Children'][child]
    else:
        return Parent[ChildName]

def findModel(Parent,PathElements):
    for pe in PathElements:
        Parent = findNextChild(Parent,pe)
    return Parent    

def StopReporting(WheatApsimx,modelPath):
    PathElements = modelPath.split('.')
    report = findModel(WheatApsimx,PathElements)
    report["EventNames"] = []

def removeModel(Parent,modelPath):
    PathElements = modelPath.split('.')
    Parent = findModel(Parent,PathElements[:-1])
    pos = 0
    found = False
    for c in Parent['Children']:
        if c['Name'] == PathElements[-1]:
            del Parent['Children'][pos]
            found = True
            break
        pos += 1
    if found == False:
        print('Failed to find ' + PathElements[-1] + ' to delete')

def ApplyParamReplacementSet(paramValues,paramNames):
    with open(WorkingLentilFilePath,'r') as WorkingLentilApsimxJSON:
        WorkingLentilApsimx = json.load(WorkingLentilApsimxJSON)
        WorkingLentilApsimxJSON.close()
    ## Remove old prameterSet manager in replacements
    #removeModel(LentilApsimx,'Replacements.SetCropParams')

    ## Add crop coefficient overwrite into replacements
    codeArray = ["using Models.Core;",
                 "using System;",
                 "namespace Models",
                 "{",
                 "   [Serializable]",
                 "   public class Script : Model",
                 "   {",
                 "      [Link] Zone zone;",
                 "      [EventSubscribe(\"Sowing\")]",
                 "      private void OnSowing(object sender, EventArgs e)",
                 "      {",
                 "         object Pval = 0;"]
    for p in range(len(paramValues)):
        codeArray.append("         Pval = " + str(paramValues[p]) + ";")
        codeArray.append('         zone.Set(\"' + paramNames[p] + '\", Pval);')
        
    codeArray.append('      }')
    codeArray.append('   }')
    codeArray.append('}')

    SetCropParams["CodeArray"] = codeArray

    AppendModeltoModelofType(WorkingLentilApsimx,"Models.Core.Folder, Models","Replacements",SetCropParams)

    with open(WorkingLentilFilePath,'w') as WorkingLentilApsimxJSON:
        json.dump(WorkingLentilApsimx,WorkingLentilApsimxJSON,indent=2)
        
def makeLongString(SimulationSet):
    longString =  '/SimulationNameRegexPattern:"'
    longString =  longString + '(' + SimulationSet[0]  + ')|' # Add first on on twice as apsim doesn't run the first in the list
    for sim in SimulationSet[:]:
        longString = longString + '(' + sim + ')|'
    longString = longString + '(' + SimulationSet[-1] + ')|' ## Add Last on on twice as apsim doesnt run the last in the list
    longString = longString + '(' + SimulationSet[-1] + ')"'
    return longString

def CalcScaledValue(Value,RMax,RMin):
    return (Value - RMin)/(RMax-RMin)
# +
def Preparefile():
    ## Empty the data store
    # !del C:\GitHubRepos\ApsimX\Prototypes\Lentil\LentilOptItter.db
    ## Add blank manager into each simulation
    with open(BaseLentilFilePath,'r') as BaseLentilApsimxJSON:
        WorkingLentilApsimx = json.load(BaseLentilApsimxJSON)
    BaseLentilApsimxJSON.close()
    AppendModeltoModelofTypeAndDeleteOldIfPresent(WorkingLentilApsimx,'Models.Core.Zone, Models',BlankManager)
    with open(WorkingLentilFilePath,'w') as WorkingLentilApsimxJSON:
        json.dump(WorkingLentilApsimx,WorkingLentilApsimxJSON,indent=2)
    #print('File Prep Complete')
    
def runModelItter(paramNames,paramValues,FittingVariables,Cultivar,ver):
    print(paramValues)
    Preparefile()
    #Put parameter values for current itteration into setcropParameters script
    ApplyParamReplacementSet(paramValues,paramNames)
    
    OptimisationVariables = ['Observed.'+x for x in FittingVariables]
    #DataPresent = pd.Series(index = DataTable.index,dtype=bool)
    #DataPresent = False
    # for v in OptimisationVariables:
    #     DataPresent = (DataPresent | ~np.isnan(pd.to_numeric(DataTable.loc[:,v])))
    SimulationSet = [BaseSimNames[x]+Cultivar for x in range(len(BaseSimNames))]
    #SimSet = makeLongString(SimulationSet)
    SimSet = '/SimulationNameRegexPattern:('+Cultivar+')'
    #print(SimSet)
    start = dt.datetime.now()
    subprocess.run(['C:/GitHubRepos/ApsimX/bin/Debug/net6.0/Models.exe',
                    WorkingLentilFilePath,
                   SimSet], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    endrun = dt.datetime.now()
    runtime = (endrun-start).seconds
    con = sqlite3.connect(DBFilePath)
    try:
        ObsPred = pd.read_sql("Select * from HarvestObsPred",con)
        Sims = pd.read_sql("Select * from _Simulations",con)
        Sims.set_index('ID',inplace=True)
        Factors = pd.read_sql("Select * from _Factors",con)
        Factors.set_index('SimulationID', inplace = True)
        con.close()
    except:
        con.close()
        
    
    ObsPred.loc[:,"SimulationName"] = [Sims.loc[ObsPred.loc[x,'SimulationID'],'Name'] for x in ObsPred.index]
    ObsPred.loc[:,"ExperimentName"] = [Factors.loc[ObsPred.loc[x,'SimulationID'],'ExperimentName'].iloc[0] for x in ObsPred.index]
    n = len(ObsPred.loc[:,'SimulationName'].drop_duplicates())        

    ScObsPre = pd.DataFrame(columns = ['ScObs','ScPred','Var','SimulationName','ExperimentName'])
    indloc = 0
    for var in FittingVariables:
        DataPairs = ObsPred.reindex(['Observed.'+var,'Predicted.'+var,'SimulationName','ExperimentName'],axis=1).dropna()
        for v in ['Observed.'+var,'Predicted.'+var]:
            DataPairs.loc[:,v] = pd.to_numeric(DataPairs.loc[:,v])
        VarMax = max(DataPairs.loc[:,'Observed.'+var].max(),DataPairs.loc[:,'Predicted.'+var].max())
        VarMin = min(DataPairs.loc[:,'Observed.'+var].min(),DataPairs.loc[:,'Predicted.'+var].min())
        for x in DataPairs.index:
            ScObsPre.loc[indloc,'ScObs'] = CalcScaledValue(DataPairs.loc[x,'Observed.'+var],VarMax,VarMin)
            ScObsPre.loc[indloc,'ScPred'] = CalcScaledValue(DataPairs.loc[x,'Predicted.'+var],VarMax,VarMin)
            ScObsPre.loc[indloc,'Var'] = var
            globals()['IttersObsPred'].loc[(Cultivar,globals()['itter'],indloc),['ScObs','ScPred','Var']] = ScObsPre.loc[indloc,['ScObs','ScPred','Var']]
            globals()['IttersObsPred'].loc[(Cultivar,globals()['itter'],indloc),
                                           ['SimulationName','ExperimentName']] = DataPairs.loc[x,['SimulationName','ExperimentName']]
            indloc+=1
    RegStats = MUte.MathUtilities.CalcRegressionStats('LN',ScObsPre.loc[:,'ScPred'].values,ScObsPre.loc[:,'ScObs'].values)
    globals()['IttersObsPred'].loc[(Cultivar,globals()['itter']),['NSE','nSims']] = [RegStats.NSE,n]
    globals()['IttersObsPred'].loc[(Cultivar,globals()['itter']),paramNames] = paramValues
    try:
        retVal = max(RegStats.NSE,-2) *-1
        print(str(globals()['itter']) + ' NSE = '+str(RegStats.NSE)+ ".  run completed " +str(n) + ' sims in '+ str(runtime) + ' seconds.')
    except:
        retVal = 2
        print(str(globals()['itter']) + ' NSE = '+str(RegStats.NSE)+ ".  run completed " +str(n) + ' sims in '+ str(runtime) + ' seconds.')
    globals()['IttersObsPred'].to_pickle("./OptFiles/"+c+"_IttersObsPred"+ver+".pkl")
    globals()['itter'] +=1
    return retVal

def runDevModelItter(paramValues):
    retVal = runModelItter(paramNames,paramValues,FittingVariables,c,Version)
    return retVal


# -

Version  ='1'
IttersObsPred = pd.DataFrame(columns = ['ScObs','ScPred','Var','Predicted.Simulation.Name','Predicted.Experiment','Predicted.Wheat.SowingDate',
                                                       'NSE','nSims']+paramNames,
                                                        index=pd.MultiIndex.from_arrays([[],[],[]],names=['Cultivar','itter','indloc']))
itter = 1
c='Precoz'
paramValues = [814.6453225472756, 0.49836726151624733, 107.03559095889537, 0.8059542544259072, 460.5773909993857, 0.23494631128887233]
runDevModelItter(paramValues)


Version = '2'
Cultivars = ['Precoz','Laird']
#Kappas=pd.Series(index=[0,1,2],data=[100,10,1])
FailedCultivars = []
rounds = 0
while (rounds<10):
    for c in Cultivars:
        print ('Fitting '+c+ ' Round ' + str(rounds))
        try:
            checkpoint_saver = CheckpointSaver("./OptFiles/"+c+" FitsCheckpoint"+Version+".pkl", compress=9)
            RandomCalls = 20
            OptimizerCalls = 5
            TotalCalls = RandomCalls + OptimizerCalls
            x0 = [100.0, 0.5, 100.0, 0.5, 100, 0.5]
            bounds = [(100.0,1000.0),
                      (0.0,1.0),
                      (0.0,500.0),
                      (0.0,1.0),
                      (100.0,500.0),
                      (0.0,1.0)]

            if rounds == 0:
                print ('Creating empty IttersObsPred dataframe')
                globals()['IttersObsPred'] = pd.DataFrame(columns = ['ScObs','ScPred','Var','SimulationName','ExperimentName','Predicted.Wheat.SowingDate',
                                                       'NSE','nSims']+paramNames,
                                                        index=pd.MultiIndex.from_arrays([[],[],[]],names=['Cultivar','itter','indloc']))
                globals()['itter'] = 0
                ret = gp_minimize(runDevModelItter, bounds, n_calls=TotalCalls,n_initial_points=RandomCalls,
                         initial_point_generator='sobol',callback=[checkpoint_saver],x0=x0)
                currentBest = ret.fun
            else:
                CheckPoint = load("./OptFiles/"+c+" FitsCheckpoint"+Version+".pkl")
                globals()['IttersObsPred'] = pd.read_pickle("./OptFiles/"+c+"_IttersObsPred"+Version+".pkl")
                globals()['itter'] = len(CheckPoint.func_vals)
                x0 = CheckPoint.x_iters
                y0 = CheckPoint.func_vals
                ret = gp_minimize(runDevModelItter, bounds, n_calls=TotalCalls,n_initial_points=RandomCalls,
                         initial_point_generator='sobol',callback=[checkpoint_saver],x0=x0,y0=y0)
                currentBest = ret.fun
            print(c)
            print(str(rounds))
            print(str(currentBest))
        except:
            FailedCultivars.append(c)
    rounds += 1

CheckPoint = load("./OptFiles/"+"Precoz"+" FitsCheckpoint"+Version+".pkl")

CheckPoint

CheckPoint

# +
ShortParams = pd.Series(index=paramNames+['NSE'],data=['JuvBase','VrnSens','IndBase','IndPpSens','ERbase','ERPpSens','NSE'])

def plotScObsPred(obsPreSet,alp,leg=True):
    markers = ['o','s','^','v','1','2','3','4','+','x','d']
    colors = pd.Series(index = ['Lentil.Phenology.StartBuddingDAS','Lentil.Phenology.StartFloweringDAS'],
                       data = ['r','b'])
    
    obsPredSet = obsPreSet.sort_values(['Var','ExperimentName']).copy()
    for var in obsPreSet.loc[:,'Var'].drop_duplicates():
        epos = 0
        if leg==True:
            lablab = var.replace('Lentil.Phenology.','')
        else:
            lablab = None
        for expt in obsPreSet.loc[:,'ExperimentName'].drop_duplicates():
            filt = ((obsPreSet.loc[:,'Var'] == var) & (obsPreSet.loc[:,'ExperimentName'] == expt))
            plt.plot(obsPreSet.loc[filt,'ScObs'],obsPreSet.loc[filt,'ScPred'],
                     markers[epos],color=colors[var],alpha=alp,label=lablab)
            
            epos+=1
            if epos == 11:
                epos= 0
            lablab = None
            
def PlotObsPre(obsPreSet,c,ax,leg=True):
    plotScObsPred(obsPreSet,1)
    plt.plot([0,1],[0,1],'-',color='k')
    plt.ylabel('Scalled Predictions')
    plt.xlabel('Scalled Observations')
    if leg==True:
        plt.legend(bbox_to_anchor=(1.1, 1.05))
    RegStats = MUte.MathUtilities.CalcRegressionStats('LN',obsPreSet.loc[:,'ScPred'].values,obsPreSet.loc[:,'ScObs'].values)
    plt.text(1.6,0.03,'new NSE = '+str(RegStats.NSE)[:4],transform=ax.transAxes,horizontalalignment='right')
    plt.text(0.03,0.97,c,horizontalalignment='left')
           
def PlotCultivar(obsPreSet,c,ret,ParamCombs):
    graph = plt.figure(figsize=(10,20))
    gs = gridspec.GridSpec(6, 6)
    ax = graph.add_subplot(gs[0, 0:2])
    PlotObsPre(obsPreSet,c,ax)

    ax = graph.add_subplot(gs[0, 4:6])
    CurrentBest = [-ParamCombs.loc[0,'NSE']]
    for x in ParamCombs.index[1:]:
        CurrentBest.append(max(-ParamCombs.loc[x,'NSE'],CurrentBest[x-1]))
    plt.plot(CurrentBest,'-',color='b')
    plt.ylabel('Best NSE achieved')
    plt.xlabel('Itteration')
    plt.ylim(-2,1)
    for r in range(0,7):
        x1 = r * 40
        x2 = r * 40 + 25
        plt.fill_between([x1,x2],1,-2,color='b',alpha=0.1)
        x1 +=25
        x2 +=15
        plt.fill_between([x1,x2],1,-2,color='yellow',alpha=0.1)
    try:
        x0 = list(CampInputs.loc[c,paramNames].values)
    except:
        x0 = [np.nan]*6
    bestFit = ParamCombs.loc[:,'NSE'].idxmin()
    adequateFits = min(sum(ParamCombs.sort_values('NSE').NSE.values<-0.5),sum(ParamCombs.sort_values('NSE').NSE.values<ParamCombs.loc[bestFit,'NSE']*.9))
    if np.isnan(x0[0]):
        BestSet.loc[c,:] = ParamCombs.loc[bestFit,:]
    else:
        try:
            CandidateSets = ParamCombs.sort_values('NSE').iloc[:adequateFits,:].copy()
            bestSetInd = CandidateSets.loc[:,'CombScore'].idxmax()
            BestSet.loc[c,:] = CandidateSets.loc[bestSetInd,:] 
        except:
            BestSet.loc[c,:] = ParamCombs.loc[bestFit,:]
            print('No acceptable fits for '+c)
    pos = 0
    for p in paramNames:
        ax = graph.add_subplot(gs[1,pos])
        plt.plot(ParamCombs.loc[:,p],-ParamCombs.loc[:,'NSE'],'o',color='k')
        plt.plot(x0[pos],max(0,-ret.func_vals[0]),'o',color='orange',ms=8,mec='k',mew=2)
        plt.plot(ParamCombs.loc[bestFit,p],-ParamCombs.loc[bestFit,'NSE'],'o',color='cyan',ms=8,mec='k',mew=2)
        plt.plot(BestSet.loc[c,p],-BestSet.loc[c,'NSE'],'o', ms=8, mfc="None",mec='m', mew=2)
        if pos == 0:
            plt.ylabel('NSE')
        else:
            ax.axes.yaxis.set_visible(False)
        plt.xlabel(ShortParams[p])
        plt.ylim(0,1)
        
        ax = graph.add_subplot(gs[2,pos])
        plt.plot(ParamCombs.sort_values('NSE').loc[:,p].values,'o',ms=1)
        plt.plot([0,adequateFits],[ParamCombs.loc[bestFit,p]]*2,'--',color='cyan',ms=8,mec='k',mew=2,lw=1)
        plt.plot([0],[ParamCombs.loc[bestFit,p]],'-',color='cyan',ms=4,mec='k',mew=2)
        initRank = ParamCombs.loc[0,'Rank']
        if np.isnan(x0[0]):
            do='nothing'
        else:
            plt.plot([0,initRank],[x0[pos]]*2,'--',color='orange',ms=8,mec='k',mew=2,lw=1)
            plt.plot([initRank],[x0[pos]],'o',color='orange',ms=4,mec='k',mew=1)
        plt.plot(BestSet.loc[c,'Rank'],BestSet.loc[c,p],'o', ms=8, mfc="None",mec='m', mew=2)
        if pos == 0:
            plt.ylabel('Parameter value')
        plt.xlabel('Rank')
        ax.tick_params(axis='y',direction='out', pad=1,rotation=90)
        ax2 = ax.twinx()
        ax2.plot(-ParamCombs.sort_values('NSE').loc[:,'NSE'].values ,'-',color='r')
        ax.set_xscale('log')
        plt.fill_between([adequateFits,len(ParamCombs.NSE.values)],1,-3,color='r',alpha=0.1)
        plt.ylim(0,1)
        if pos == 5:
            plt.ylabel('NSE')
        else:
            ax2.axes.yaxis.set_ticklabels([])
        pos+=1
        


# -


#Version = '5'
BestSet = pd.DataFrame(columns = paramNames+['NSE','Rank','InitAgree','CombScore'])
for c in Cultivars:
    IttersObsPred = pd.read_pickle("./OptFiles/"+c+"_IttersObsPred"+Version+".pkl")
    ret = load("./OptFiles/"+c+" FitsCheckpoint"+Version+".pkl")
    ParamCombs = pd.DataFrame(ret.x_iters,columns = paramNames)
    ParamCombs.loc[:,'NSE'] = ret.func_vals
    bestFit = ParamCombs.loc[:,'NSE'].idxmin()
    ParamCombs.sort_values('NSE',inplace=True)
    ParamCombs.loc[:,'Rank'] = range(len(ParamCombs.index))
    ParamCombs.loc[:,'InitAgree'] = np.nan
    initRank = ParamCombs.loc[0,'Rank']
    for x in ParamCombs.index[:initRank+1]:
        ParamCombs.loc[x,'InitAgree'] = MUte.MathUtilities.CalcRegressionStats('AS',ParamCombs.loc[0,:][:-5].values,ParamCombs.loc[x,:][:-5].values).NSE
    adequateFits = min(sum(ParamCombs.NSE.values<-0.5),sum(ParamCombs.NSE.values<ParamCombs.loc[bestFit,'NSE']*.8))
    ParamCombs.loc[:,'CombScore']= -ParamCombs.loc[:,'NSE']*ParamCombs.loc[:adequateFits,'InitAgree']
    ParamCombs.sort_values('CombScore',ascending=False,inplace=True)
    # bestSetInd = ParamCombs.loc[:,'CombScore'].idxmax()
    # BestSet.loc[c,:] = ParamCombs.loc[bestSetInd,:] 
    ParamCombs.sort_index(inplace=True)
    bestFitItter = ParamCombs.loc[:,'NSE'].idxmin()
    bestFitObsPred = IttersObsPred.loc[(c,bestFitItter),:]
    PlotCultivar(bestFitObsPred,c,ret,ParamCombs)
    #ParamCombs.sort_values('CombScore',ascending=False,inplace=True)

BestSet.transpose()
