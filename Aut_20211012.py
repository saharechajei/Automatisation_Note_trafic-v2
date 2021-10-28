import streamlit as st
from dateutil.parser import parse
import datetime
#import turtle
#import tkinter as TK

# import matplotlib.patches as patches
#import matplotlib as mpl
#from matplotlib.collections import PatchCollection
import matplotlib as plt
#import seaborn as sns
import numpy as np
import pandas as pd
from pylab import *
from sklearn.preprocessing import StandardScaler
from heapq import nsmallest, nlargest
from operator import itemgetter

st.write('''
# *Note mensuelle*

''')
#Hello *world*
st.sidebar.header("Les paramètres d'entrée")
#def user_input():
Année_Select = st.sidebar.number_input('Année sélectionnée',value=0,key=0)
Mois_Select=st.sidebar.number_input('Mois sélectionné',value=0,key=1)
Année_Select: int=int(Année_Select)
Mois_Select=int(Mois_Select)
#    return (Année_Select, Mois_Select)
#df0=user_input()
#st.write(df0)

#    int(input('Veuillez entrer une année: '))
data = pd.read_excel('Trafic portruaire_Juillet2021.xlsx')

df = data.copy()
#df.head()
#df.shape
#st.write(df.shape)
#st.write(df.head())
# st.write(df.shape)


# In[3]:


# X={0 if df['Trafic'].unique()=='Ciment'else 1}
##st.write(list(df['Sous_Catégorie'].unique()))


# In[4]:



# In[5]:


# les séries Ai doivent être automatisées
A1 = {"20' Plein", "20' Vide", "40' Plein", "40' Vide", "45' Plein", "45' Vide", "TC-Tonnage"}
A2 = {"20' Plein", "20' Vide"}
A3 = {"40' Plein", "40' Vide"}
A4 = {"45' Plein", "45' Vide"}
A5 = {"20' Plein", "40' Plein", "45' Plein"}
A6 = {"20' Vide", "40' Vide", "45' Vide"}
A7 = {"Croisière", "Sortie", "Entrée", "Cabotage-Export", "Export", "Import", "Cabotage-Import"}
A8 = {"Ensemble routier plein", "Ensemble routier vide", "Remorque Pleinne", "Remorque Vide", "TIR", "TIR plein",
      "TIR vide"}
A9 = {"Autre Produits sidérurgiques", "Billettes", "Bobines tôle", "Fer à béton", "Fer en fradeaux", "Fil machine",
      "Slaps"}
A10 = {"Autre Bois & dérivés", "Bois de sciage", "Fardeaux de bois", "Grumes de bois", "Pate à papier",
       "Pate cellulose", "Poteaux de bois", "Rondins de bois", "Rouleaux de papier", "Vieux papier"}
A11 = {"A. Hydrocarbures", "Bitume", "Butane", "Essence", "Ethylène", "Fuel Oil", "Gasoil", "Huile de base", "Jet",
       "KEROSENE", "Pétrole brut", "Propane", "Virgin Naphta"}
A12 = {"Engrais", "Engrais en Sacs"}
A13 = {"Soufre", "Soufre Liquide"}
df['Marchandises_V0'] = df.apply(lambda x: 'Conteneurs' if x['Trafic'] in A1 else x['Trafic'], axis=1)
df['Marchandises'] = df.apply(lambda x: 'Conteneurs' if x['Trafic'] in A1 else ('TIR' if x['Trafic'] in A8 else (
    'Produits sidérurgiques' if x['Trafic'] in A9 else ("Bois & dérivés" if x['Trafic'] in A10 else (
        'Hydrocarbures' if x['Trafic'] in A11 else (
            "Engrais" if x['Trafic'] in A12 else ('Soufre' if x['Trafic'] in A13 else x['Trafic'])))))), axis=1)

# st.write(df['MarchandisesV3'].value_counts())
df['Dimension_Conteneurs'] = df.apply(lambda x: '20 Pieds' if x['Trafic'] in A2 else (
    '40 Pieds' if x['Trafic'] in A3 else ('45 Pieds' if x['Trafic'] in A4 else x['Trafic'])), axis=1)
# df['Dimension_Conteneurs']=df.apply(lambda x:'Conteneurs' if x['Trafic'] in BB else x['Trafic'], axis=1 )
# st.write(df['Dimension_Conteneurs'].value_counts())
df['Type_Conteneurs'] = df.apply(
    lambda x: 'Plein' if x['Trafic'] in A5 else ('Vide' if x['Trafic'] in A6 else x['Trafic']), axis=1)
# st.write(df['Type_Conteneurs'].value_counts())
df['Sens_ImpExp'] = df.apply(lambda x: 'Import' if x['Sens'] in {"Import", "Cabotage-Import"} else (
    'Export' if x['Sens'] in {"Cabotage-Export", "Export"} else x['Sens']), axis=1)
# st.write(df['Sens_ImpExp'].value_counts())
df['Sens_ImpExpCab'] = df.apply(
    lambda x: 'Cabotage' if x['Sens'] in {"Cabotage-Import", "Cabotage-Export"} else x['Sens'], axis=1)
# st.write(df['Sens_ImpExpCab'].value_counts())
df['Année'] = df.Mois.dt.year
df['Mois'] = df.Mois.dt.month
df['Port_v0'] = df['Port']
df['Port'] = df.apply(lambda x: 'Les ports de Safi' if x['Port'] in {'Safi Atlantique', 'Safi'} else x['Port'], axis=1)
##st.write(df.Port.unique())
##st.write(df.Port_v0.unique())
##df.head()


# In[6]:


##st.write(df.columns)
##df.dtypes


# In[7]:


df['Valeurs'] = df[
    ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
     '2019', '2020', '2021']].sum(axis=1)

##df.head()


# In[8]:


df['Combined'] = df.apply(lambda x: '%s%s%s%s%s%s%s%s%s%s%s%s' % (
x['Conditionnement'], x['Trafic'], x['Marchandises_V0'], x['Dimension_Conteneurs'], x['Type_Conteneurs'], x['Sens'],
x['Sens_ImpExp'], x['Sens_ImpExpCab'], x['Port_v0'], x['Opérateur'], x['Unité'], x['Sous_Catégorie']), axis=1)

Comb1 = {
    "Divers ConventionnelEngrais en SacsEngrais en SacsEngrais en SacsEngrais en SacsExportExportExportJorf LasfarMarsa MarocTonneBig Bag & sacherie Divers",
    "Divers ConventionnelEngrais en SacsEngrais en SacsEngrais en SacsEngrais en SacsExportExportExportJorf LasfarOCPTonneBig Bag & sacherie Divers",
    "Vrac LiquidesAcide phosphoriqueAcide phosphoriqueAcide phosphoriqueAcide phosphoriqueExportExportExportJorf LasfarOCPTonneProduits chimiques",
    "Vrac LiquidesAcide phosphoriqueAcide phosphoriqueAcide phosphoriqueAcide phosphoriqueExportExportExportSafiOCPTonneProduits chimiques",
    "Vrac LiquidesAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueExportExportExportJorf LasfarOCPTonneProduits chimiques",
    "Vrac LiquidesAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueImportImportImportJorf LasfarOCPTonneProduits chimiques",
    "Vrac LiquidesAmmoniacAmmoniacAmmoniacAmmoniacExportExportExportJorf LasfarOCPTonneProduits chimiques",
    "Vrac LiquidesAmmoniacAmmoniacAmmoniacAmmoniacImportImportImportJorf LasfarOCPTonneProduits chimiques",
    "Vrac LiquidesSoude caustiqueSoude caustiqueSoude caustiqueSoude caustiqueImportImportImportJorf LasfarOCPTonneProduits chimiques",
    "Vrac LiquidesSoufre LiquideSoufre LiquideSoufre LiquideSoufre LiquideImportImportImportJorf LasfarOCPTonneProduits chimiques",
    "Vrac SolidesEngraisEngraisEngraisEngraisExportExportExportSafiMarsa MarocTonneAutres",
    "Vrac SolidesEngraisEngraisEngraisEngraisExportExportExportJorf LasfarOCPTonneAutres",
    "Vrac SolidesEngraisEngraisEngraisEngraisExportExportExportSafiOCPTonneAutres",
    "Vrac SolidesPhosphatePhosphatePhosphatePhosphateExportExportExportCasablancaOCPTonneMinerais",
    "Vrac SolidesPhosphatePhosphatePhosphatePhosphateExportExportExportJorf LasfarOCPTonneMinerais",
    "Vrac SolidesPhosphatePhosphatePhosphatePhosphateExportExportExportLaayouneOCPTonneMinerais",
    "Vrac SolidesPhosphatePhosphatePhosphatePhosphateExportExportExportSafiOCPTonneMinerais",
    "Vrac SolidesSoufreSoufreSoufreSoufreImportImportImportSafiMarsa MarocTonneMinerais",
    "Vrac SolidesSoufreSoufreSoufreSoufreImportImportImportJorf LasfarOCPTonneMinerais"}

Comb2 = {"Vrac SolidesSoufreSoufreSoufreSoufreImportImportImportCasablancaMarsa MarocTonneMinerais",
         "Vrac SolidesSoufreSoufreSoufreSoufreImportImportImportJorf LasfarMarsa MarocTonneMinerais",
         "Vrac SolidesSoufreSoufreSoufreSoufreImportImportImportCasablancaSOMAPORTTonneMinerais"}

Comb3 = {
    "Divers ConventionnelEngrais en SacsEngrais en SacsEngrais en SacsEngrais en SacsExportExportExportSafiMarsa MarocTonneBig Bag & sacherie Divers",
    "Vrac LiquidesAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueImportImportImportAgadirMarsa MarocTonneProduits chimiques",
    "Vrac LiquidesAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueImportImportImportMohammediaMarsa MarocTonneProduits chimiques",
    "Vrac LiquidesAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueAcide sulfiriqueImportImportImportSafiMarsa MarocTonneProduits chimiques",
    "Vrac SolidesEngraisEngraisEngraisEngraisExportExportExportJorf LasfarMarsa MarocTonneAutres",
    "Vrac SolidesEngraisEngraisEngraisEngraisExportExportExportNadorMarsa MarocTonneAutres"}

df['Ind_PhosphatesDerives'] = df.apply(
    lambda x: 1 if x['Combined'] in Comb1 else (2 if x['Combined'] in Comb2 else (3 if x['Combined'] in Comb3 else 0)),
    axis=1)
##df.head()
# st.write(df.loc[df.Combinedisin(comb3)])


# In[52]:


# Bloc 1:Trafic global
Mois1 = ['JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN', 'JUILLET', 'AOUT', 'SEPTEMBRE', 'OCTOBRE', 'NOVEMBRE',
         'DECEMBRE']
Mois2 = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'September', 'Octobre', 'Novembre',
         'Décembre']
Cumul_Mois = [' pour le premier mois ', ' pour les deux premiers mois', ' pour le premier trimestre',
              ' pour les quatre premiers mois', ' pour les cinq premiers mois', ' pour le premier semestre',
              ' pour les sept premier mois', ' pour Les huit premiers mois', ' pour les trois premiers trimestres',
              ' pour les dix premiers mois', ' pour les onze premiers mois', ' ']

#Année_Select = int(input('Veuillez entrer une année: '))
#Mois_Select = int(input('Veuillez entrer un mois: '))
#Année_Select, Mois_Select = user_input()
st.write('''* ACTIVITE PORTUAIRE FIN''', Mois1[Mois_Select - 1], Année_Select)
Tr_Glob_i1 = 0
Tr_Glob_i2 = 0
Tr_Glob_i3 = 0
Tr_Glob_i4 = 0
Tr_Glob_i5 = 0
i = 0
for i in range(Mois_Select + 1):
    Tr_Glob_i1 = Tr_Glob_i1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne")].sum(axis=0)
    Tr_Glob_i2 = Tr_Glob_i2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne")].sum(axis=0)
    Tr_Glob_i3 = Tr_Glob_i3 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 2)) & (df.Mois == i) & (df.Unité == "Tonne")].sum(axis=0)
    Tr_Glob_i4 = Tr_Glob_i4 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 3)) & (df.Mois == i) & (df.Unité == "Tonne")].sum(axis=0)
    Tr_Glob_i5 = Tr_Glob_i5 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 4)) & (df.Mois == i) & (df.Unité == "Tonne")].sum(axis=0)

    i = i + 1
Taux_ann = (Tr_Glob_i1 - Tr_Glob_i2) / Tr_Glob_i2
Taux_ann_abs = abs((Tr_Glob_i1 - Tr_Glob_i2) / Tr_Glob_i2)
# st.write(Tr_Glob_i1,Tr_Glob_i2, '{0:.2%}'.format(Taux_ann))

if Taux_ann < 0:
    st.write('Un volume global de ', round((Tr_Glob_i1 / 1000000), 1), 'MT, en baisse de ', '{0:.1%}'.format(Taux_ann_abs))
    st.write('Le trafic transitant par les ports gérés par l’ANP a atteint à fin', Mois2[Mois_Select - 1], Année_Select,
          'un volume global de ', round((Tr_Glob_i1 / 1000000), 1), ' millions de tonnes, soit une baisse de ',
          '{0:.1%}'.format(Taux_ann_abs), 'en glissement annuel.')
else:
    st.write('Un volume global de ', round((Tr_Glob_i1 / 1000000), 1), 'MT, en hausse de ', '{0:.1%}'.format(Taux_ann_abs))
    st.write('Le trafic transitant par les ports gérés par l’ANP a atteint à fin', Mois2[Mois_Select - 1], Année_Select,
          'un volume global de ', round((Tr_Glob_i1 / 1000000), 1), ' millions de tonnes, soit une hausse de ',
          '{0:.1%}'.format(Taux_ann_abs), 'en glissement annuel.')

# Représentation graphique Evolution des cinq dernières années
st.write('Le graphique ci-après présente l’évolution de l’activité des ports gérés par l’ANP',
         Cumul_Mois[Mois_Select - 1], 'des cinq dernières années.')

x1 = [(Année_Select - 4), (Année_Select - 3), (Année_Select - 2), (Année_Select - 1), (Année_Select)]
y1 = [round((Tr_Glob_i5 / 1000)), round((Tr_Glob_i4 / 1000)), round((Tr_Glob_i3 / 1000)), round((Tr_Glob_i2 / 1000)),
      round((Tr_Glob_i1 / 1000))]
# x1={str(Année_Select),str(Année_Select-1),str(Année_Select-2),str(Année_Select-3),str(Année_Select-4)}
taux_5 = '{0:.1%}'.format(((pow((Tr_Glob_i1 / Tr_Glob_i5), (1 / 5))) - 1))
##st.write(x1)
##st.write(y1)
st.write('taux d’évolution:', taux_5)
#plt.figure(figsize=(10, 7))
fig, ax = plt.subplots(figsize=(7,3.5))
# ylim(30000.0,100000.0)
ax.bar(x1, y1, color='blue')
# Annotating the bar plot with the values (total)


for i in range(len(y1)):
#    plt.annotate(str(y1[i]), xy=(x1[i], y1[i]), ha='center', size=5, xytext=(0, 9), textcoords="offset points")
    plt.annotate(str(y1[i]), xy=(x1[i], y1[i]), ha='center', xytext=(0, 9), textcoords="offset points")

# plt.title('taux d’évolution', **taux_5)
plt.ylim((0 if (min(y1) - 15000) < 5000 else 15000), (max(y1) + 5000))
plt.ylabel('En milliers de tonnes')
st.pyplot(fig)

# In[53]:


##st.write(len(df['Année'].unique()))
##st.write(df.Année[100])
A = [(Année_Select - 1), (Année_Select)]
df_Select = df.loc[df.Année.isin([(Année_Select - 1), (Année_Select)])]
df_Select.head()
Table_Select = (pd.pivot_table(df_Select,
                               index=['Port', 'Sous_Catégorie', 'Trafic', 'Marchandises', 'Marchandises_V0', 'Combined',
                                      'Ind_PhosphatesDerives', 'Opérateur', 'Unité', 'Sens', 'Sens_ImpExp',
                                      'Sens_ImpExpCab', 'Mois'], columns=['Année'], values='Valeurs', aggfunc=np.sum,
                               fill_value=0).reset_index())

# Table_Select=pd.pivot_table(df_Select ,index=[ 'Marchandises','Mois'], columns=['Année'],  values='Valeurs',aggfunc= ["sum"],fill_value=0)
# Table_Select=pd.pivot_table(df_Select ,index=['Trafic', 'Sens_ImpExpCab','Mois'], columns=['Année'],  values='Valeurs',aggfunc= np.sum,fill_value=0)
# st.write(Table_Select[2021])
# st.write(Table_Select[('sum','%s'% (Année_Select))])
# Table_Select=Table_Select.ix[:,[('sum',(Année_Select)),('sum',(Année_Select-1))]]
# Table_Select.columns[(Année_Select),(Année_Select-1)]
# Table_Select['Value_Diff']=Table_Select.apply(lambda x: (x['Année'].loc[(x.Année==Année_Select)])-(x['Année'].loc[(x.Année==(Année_Select-1))]))

Table_Select['Value_Diff'] = Table_Select.apply(lambda x: (x[Année_Select] - x[(Année_Select - 1)]), axis=1)

# st.write(df.columns)
# st.write(Table_Select.columns)

##Table_Select.head()


# st.write(Table_Select[2021].loc[(Table_Select.Mois==1)&(Table_Select.Marchandises=='Conteneurs')].sum(axis=0))


# In[54]:


# Principales variations des trafics _
# Export
list_march_VarExp = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].unique()):
    list_march_VarExp[j] = 0
    march_exp = 0
    for i in range(Mois_Select + 1):
        march_exp = march_exp + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Export")].sum(axis=0)
        i = i + 1
    list_march_VarExp[j] = march_exp
##st.write(list_march_VarExp)
# Fonction de tri
# list_march_VarExp_tri=sorted(list_march_VarExp.items(), key=lambda t: t[1])
# st.write(list_march_VarExp_tri)


# from heapq import nsmallest, nlargest
# from operator import itemgetter


# nlargest(n, iterable, key=None)
# get the 3 smallest members
smallest3_Exp = dict(nsmallest(3, list_march_VarExp.items(), itemgetter(1)))
largest3_Exp = dict(nlargest(3, list_march_VarExp.items(), itemgetter(1)))
##st.write(smallest3_Exp)
##st.write(largest3_Exp)
##st.write(type(largest3_Exp))
##st.write(list(smallest3_Exp.keys()))
# st.write(Table_Select[Année_Select])
# Calculer les taux de variations des principaux trafics en pourcentage____variations négatives
list_march_exp_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_Exp.keys()):
    list_march_exp_PrNeg[j] = 0
    march_exp_PrNeg1 = 0
    march_exp_PrNeg2 = 0
    for i in range(Mois_Select + 1):
        march_exp_PrNeg1 = march_exp_PrNeg1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Export")].sum(axis=0)
        march_exp_PrNeg2 = march_exp_PrNeg2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Export")].sum(axis=0)
        i = i + 1
    march_exp_PrNeg = '{0:.1%}'.format((march_exp_PrNeg1 - march_exp_PrNeg2) / march_exp_PrNeg2)
    # st.write(march_exp_PrNeg1)
    list_march_exp_PrNeg[j] = march_exp_PrNeg
##st.write(list_march_exp_PrNeg)

# Calculer les taux de variations des principaux trafics en pourcentage____variations positives
list_march_exp_PrPos = dict()
i = 0
j = 0

for j in list(largest3_Exp.keys()):
    list_march_exp_PrPos[j] = 0
    march_exp_PrPos1 = 0
    march_exp_PrPos2 = 0
    for i in range(Mois_Select + 1):
        march_exp_PrPos1 = march_exp_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Export")].sum(axis=0)
        march_exp_PrPos2 = march_exp_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Export")].sum(axis=0)
        i = i + 1
    march_exp_PrPos = '{0:.1%}'.format((march_exp_PrPos1 - march_exp_PrPos2) / march_exp_PrPos2)
    # st.write(march_exp_PrNeg1)
    list_march_exp_PrPos[j] = march_exp_PrPos
##st.write(list_march_exp_PrPos)
# st.write(list_march_exp_PrPos[1].key())

##st.write(list_march_exp_PrNeg.items())
[(ax1, bx1), (cx1, dx1), (ex1, fx1)] = list_march_exp_PrNeg.items()
##st.write(ax1)
[(ax2, bx2), (cx2, dx2), (ex2, fx2)] = list_march_exp_PrPos.items()
##st.write(list_march_exp_PrPos.items())
##st.write(dx2)


# In[55]:


# Principales variations des trafics _
# Import
list_march_VarImp = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].unique()):
    list_march_VarImp[j] = 0
    march_Imp = 0
    for i in range(Mois_Select + 1):
        march_Imp = march_Imp + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Import")].sum(axis=0)
        i = i + 1
    list_march_VarImp[j] = march_Imp
##st.write(list_march_VarImp)
# Fonction de tri
# list_march_VarImp_tri=sorted(list_march_VarImp.items(), key=lambda t: t[1])
# st.write(list_march_VarImp_tri)


# from heapq import nsmallest, nlargest
# from operator import itemgetter


# nlargest(n, iterable, key=None)
# get the 3 smallest members
smallest3_Imp = dict(nsmallest(3, list_march_VarImp.items(), itemgetter(1)))
largest3_Imp = dict(nlargest(3, list_march_VarImp.items(), itemgetter(1)))
##st.write(smallest3_Imp)
##st.write(largest3_Imp)
##st.write(type(largest3_Imp))
##st.write(list(smallest3_Imp.keys()))
# st.write(Table_Select[Année_Select])
# Calculer les taux de variations des principaux trafics en pourcentage____variations négatives
list_march_Imp_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_Imp.keys()):
    list_march_Imp_PrNeg[j] = 0
    march_Imp_PrNeg1 = 0
    march_Imp_PrNeg2 = 0
    for i in range(Mois_Select + 1):
        march_Imp_PrNeg1 = march_Imp_PrNeg1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Import")].sum(axis=0)
        march_Imp_PrNeg2 = march_Imp_PrNeg2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Import")].sum(axis=0)
        i = i + 1
    march_Imp_PrNeg = '{0:.1%}'.format((march_Imp_PrNeg1 - march_Imp_PrNeg2) / march_Imp_PrNeg2)
    # st.write(march_Imp_PrNeg1)
    list_march_Imp_PrNeg[j] = march_Imp_PrNeg
##st.write(list_march_Imp_PrNeg)

# Calculer les taux de variations des principaux trafics en pourcentage____variations positives
list_march_Imp_PrPos = dict()
i = 0
j = 0

for j in list(largest3_Imp.keys()):
    list_march_Imp_PrPos[j] = 0
    march_Imp_PrPos1 = 0
    march_Imp_PrPos2 = 0
    for i in range(Mois_Select + 1):
        march_Imp_PrPos1 = march_Imp_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Import")].sum(axis=0)
        march_Imp_PrPos2 = march_Imp_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Import")].sum(axis=0)
        i = i + 1
    march_Imp_PrPos = '{0:.1%}'.format((march_Imp_PrPos1 - march_Imp_PrPos2) / march_Imp_PrPos2)
    # st.write(march_Imp_PrNeg1)
    list_march_Imp_PrPos[j] = march_Imp_PrPos
##st.write(list_march_Imp_PrPos)
# st.write(list_march_Imp_PrPos[1].key())

##st.write(list_march_Imp_PrNeg.items())
# [(am1,bm1),(cm1,dm1),(em1,fm1), (gm1,hm1)]=list_march_Imp_PrNeg.items()

[(am1, bm1), (cm1, dm1), (em1, fm1)] = list_march_Imp_PrNeg.items()
##st.write(am1)
# [(am2,bm2),(cm2,dm2),(em2,fm2),(gm2,hm2)]=list_march_Imp_PrPos.items()

[(am2, bm2), (cm2, dm2), (em2, fm2)] = list_march_Imp_PrPos.items()
##st.write(list_march_Imp_PrPos.items())
##st.write(dm2)


# In[56]:


# Principales variations des trafics _
# Cabotage
list_march_VarCab = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].unique()):
    list_march_VarCab[j] = 0
    march_Cab = 0
    for i in range(Mois_Select + 1):
        march_Cab = march_Cab + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
        i = i + 1
    list_march_VarCab[j] = march_Cab
##st.write(list_march_VarCab)
# Fonction de tri
# list_march_VarCab_tri=sorted(list_march_VarCab.items(), key=lambda t: t[1])
# st.write(list_march_VarCab_tri)


# from heapq import nsmallest, nlargest
# from operator import itemgetter


# nlargest(n, iterable, key=None)
# get the 3 smallest members
smallest3_Cab = dict(nsmallest(3, list_march_VarCab.items(), itemgetter(1)))
largest3_Cab = dict(nlargest(3, list_march_VarCab.items(), itemgetter(1)))
##st.write(smallest3_Cab)
##st.write(largest3_Cab)
##st.write(type(largest3_Cab))
##st.write(list(smallest3_Cab.keys()))
# st.write(Table_Select[Année_Select])
# Calculer les taux de variations des principaux trafics en pourcentage____variations négatives
list_march_Cab_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_Cab.keys()):
    list_march_Cab_PrNeg[j] = 0
    march_Cab_PrNeg1 = 0
    march_Cab_PrNeg2 = 0
    for i in range(Mois_Select + 1):
        march_Cab_PrNeg1 = march_Cab_PrNeg1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
        march_Cab_PrNeg2 = march_Cab_PrNeg2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
        i = i + 1
    march_Cab_PrNeg = '{0:.1%}'.format((march_Cab_PrNeg1 - march_Cab_PrNeg2) / march_Cab_PrNeg2)
    # st.write(march_Cab_PrNeg1)
    list_march_Cab_PrNeg[j] = march_Cab_PrNeg
##st.write(list_march_Cab_PrNeg)

# Calculer les taux de variations des principaux trafics en pourcentage____variations positives
list_march_Cab_PrPos = dict()
i = 0
j = 0

for j in list(largest3_Cab.keys()):
    list_march_Cab_PrPos[j] = 0
    march_Cab_PrPos1 = 0
    march_Cab_PrPos2 = 0
    for i in range(Mois_Select + 1):
        march_Cab_PrPos1 = march_Cab_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
        march_Cab_PrPos2 = march_Cab_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
        i = i + 1
    march_Cab_PrPos = '{0:.1%}'.format((march_Cab_PrPos1 - march_Cab_PrPos2) / march_Cab_PrPos2)
    # st.write(march_Cab_PrNeg1)
    list_march_Cab_PrPos[j] = march_Cab_PrPos
##st.write(list_march_Cab_PrPos)
# st.write(list_march_Cab_PrPos[1].key())

##st.write(list_march_Cab_PrNeg.items())
# [(ac1,bc1),(cc1,dc1),(ec1,fc1), (gc1,hc1)]=list_march_Cab_PrNeg.items()

[(ac1, bc1), (cc1, dc1), (ec1, fc1)] = list_march_Cab_PrNeg.items()
##st.write(ac1)
# [(ac2,bc2),(cc2,dc2),(ec2,fc2),(gc2,hc2)]=list_march_Cab_PrPos.items()

[(ac2, bc2), (cc2, dc2), (ec2, fc2)] = list_march_Cab_PrPos.items()
##st.write(list_march_Cab_PrPos.items())
##st.write(dc2)


# In[57]:


# Principales variations des trafics _
# Cabotage
# Calcul provisoire
# march_Cab_Conteneurs=0
# march_Cab_Hydrocarb=0
# i=0
# for i in range(Mois_Select+1):
#    march_Cab_Conteneurs=march_Cab_Conteneurs+Table_Select['Value_Diff'].loc[(Table_Select.Mois==i)&(Table_Select.Marchandises=="Conteneurs")&(Table_Select.Unité=="Tonne")&(Table_Select.Sens_ImpExpCab=="Cabotage")].sum(axis=0)
#    march_Cab_Hydrocarb=march_Cab_Hydrocarb+Table_Select['Value_Diff'].loc[(Table_Select.Mois==i)&(Table_Select.Marchandises=="Hydrocarbures")&(Table_Select.Unité=="Tonne")&(Table_Select.Sens_ImpExpCab=="Cabotage")].sum(axis=0)

#    i=i+1


# st.write(Table_Select[Année_Select])
# Calculer les taux de variations des principaux trafics en pourcentage____variations négatives
i = 0
march_Cab_Conteneurs1 = 0
march_Cab_Conteneurs2 = 0
march_Cab_Hydrocarb1 = 0
march_Cab_Hydrocarb2 = 0
for i in range(Mois_Select + 1):
    march_Cab_Conteneurs1 = march_Cab_Conteneurs1 + Table_Select[Année_Select].loc[
        (Table_Select.Mois == i) & (Table_Select.Marchandises == "Conteneurs") & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
    march_Cab_Conteneurs2 = march_Cab_Conteneurs2 + Table_Select[(Année_Select - 1)].loc[
        (Table_Select.Mois == i) & (Table_Select.Marchandises == "Conteneurs") & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
    march_Cab_Hydrocarb1 = march_Cab_Hydrocarb1 + Table_Select[Année_Select].loc[
        (Table_Select.Mois == i) & (Table_Select.Marchandises == "Hydrocarbures") & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
    march_Cab_Hydrocarb2 = march_Cab_Hydrocarb2 + Table_Select[(Année_Select - 1)].loc[
        (Table_Select.Mois == i) & (Table_Select.Marchandises == "Hydrocarbures") & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Sens_ImpExpCab == "Cabotage")].sum(axis=0)

    i = i + 1

MCC0 = (march_Cab_Conteneurs1 - march_Cab_Conteneurs2) / march_Cab_Conteneurs2
MCH0 = (march_Cab_Hydrocarb1 - march_Cab_Hydrocarb2) / march_Cab_Hydrocarb2
MCC = '{0:.1%}'.format(MCC0)
MCH = '{0:.1%}'.format(MCH0)

##st.write('Hydrocarbures, ', MCH)
##st.write('Conteneurs, ', MCC)


# Calculer les taux de variations des principaux trafics en pourcentage____variations positives


# In[58]:


# Bloc TRAFIC PAR NATURE DE FLUX
st.write('TRAFIC PAR NATURE DE FLUX')
st.write('Par nature de flux, les évolutions enregistrées se présentent comme suit:')

Tr_Glob_Exp1 = 0
Tr_Glob_Exp2 = 0
i = 0
for i in range(Mois_Select + 1):
    Tr_Glob_Exp1 = Tr_Glob_Exp1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Sens_ImpExpCab == "Export")].sum(
        axis=0)
    Tr_Glob_Exp2 = Tr_Glob_Exp2 + df['Valeurs'].loc[(df.Année == (Année_Select - 1)) & (df.Mois == i) &
                                                    (df.Unité == "Tonne") & (df.Sens_ImpExpCab == "Export")].sum(axis=0)

    i = i + 1
Taux_ann_Exp = (Tr_Glob_Exp1 - Tr_Glob_Exp2) / Tr_Glob_Exp2
# st.write(Taux_ann_Exp)
if Taux_ann_Exp < 0:
    st.write('- Un recul des exportations (', '{0:.1%}'.format(Taux_ann_Exp), '), avec un volume de',
             round(Tr_Glob_Exp1 / 1000000, 1), 'MT, expliqué d’une part par la baisse des trafics suivants:', ax1, '(',
             bx1, '),', cx1, '(', dx1, ') et ', ex1, '(', fx1,
             ') et d’autre part par l’augmentation des exportations des:', ax2, '(', bx2, '),', cx2, '(', dx2, ') et ',
             ex2, '(', fx2, ').')
else:
    st.write('- Une augmentation des exportations (', '{0:.1%}'.format(Taux_ann_Exp), '), avec un volume de',
             round(Tr_Glob_Exp1 / 1000000, 1), 'MT, expliquée d’une part par la hausse des trafics:', ax2, '(', bx2,
             '),', cx2, '(', dx2, ') et ', ex2, '(', fx2, ') et d’autre part par le recul des exportations des: ', ax1,
             '(', bx1, '),', cx1, '(', dx1, ') et ', ex1, '(', fx1, ').')

Tr_Glob_Imp1 = 0
Tr_Glob_Imp2 = 0
i = 0
for i in range(Mois_Select + 1):
    Tr_Glob_Imp1 = Tr_Glob_Imp1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Sens_ImpExpCab == "Import")].sum(
        axis=0)
    Tr_Glob_Imp2 = Tr_Glob_Imp2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (
                    df.Sens_ImpExpCab == "Import")].sum(axis=0)

    i = i + 1
Taux_ann_Imp = (Tr_Glob_Imp1 - Tr_Glob_Imp2) / Tr_Glob_Imp2
# st.write(Taux_ann_Imp)
if Taux_ann_Imp < 0:
    st.write('- Une baisse des importations (', '{0:.1%}'.format(Taux_ann_Imp), '), avec un volume de',
             round(Tr_Glob_Imp1 / 1000000, 1), 'MT, expliquée d’une part par le recul des trafics suivants:', am1, '(',
             bm1, '),', cm1, '(', dm1, ') et ', em1, '(', fm1, ') et d’autre part par la hausse des importations des:',
             am2, '(', bm2, '),', cm2, '(', dm2, ') et ', em2, '(', fm2, ').')
else:
    st.write('- Une hausse des importations (', '{0:.1%}'.format(Taux_ann_Imp), '), avec un volume de',
             round(Tr_Glob_Imp1 / 1000000, 1), 'MT,expliquée d’une part par l’augmentation des trafics:', am2, '(', bm2,
             '),', cm2, '(', dm2, ') et ', em2, '(', fm2, ') et d’autre part par la baisse des importations des:', am1,
             '(', bm1, '),', cm1, '(', dm1, ') et ', em1, '(', fm1, ').')

Tr_Glob_Cab1 = 0
Tr_Glob_Cab2 = 0
i = 0
for i in range(Mois_Select + 1):
    Tr_Glob_Cab1 = Tr_Glob_Cab1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Sens_ImpExpCab == "Cabotage")].sum(
        axis=0)
    Tr_Glob_Cab2 = Tr_Glob_Cab2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (
                    df.Sens_ImpExpCab == "Cabotage")].sum(axis=0)

    i = i + 1
Taux_ann_Cab = (Tr_Glob_Cab1 - Tr_Glob_Cab2) / Tr_Glob_Cab2
# st.write(Taux_ann_Cab)
if Taux_ann_Cab < 0:
    if (MCC0 < 0) & (MCH0 < 0):
        st.write('- Une régression du cabotage (', '{0:.1%}'.format(Taux_ann_Cab), '), avec un volume de',
                 round(Tr_Glob_Cab1 / 1000000, 1),
                 'MT, induite essentiellement par la baisse du trafic des conteneurs (', MCC, ')et des hydrocarbures (',
                 MCH, ').')
    elif (MCC0 < 0) & (MCH0 > 0):
        st.write('- Une régression du cabotage (', '{0:.1%}'.format(Taux_ann_Cab), '), avec un volume de',
                 round(Tr_Glob_Cab1 / 1000000, 1),
                 'MT, induite essentiellement par la baisse du trafic des conteneurs ('
                 , MCC, '). Le trafic des hydrocarbures a, par contre, enregistré une hausse de ', MCH, '.')

    elif (MCC0 > 0) & (MCH0 < 0):
        st.write('- Une régression du cabotage (', '{0:.1%}'.format(Taux_ann_Cab), '), avec un volume de',
                 round(Tr_Glob_Cab1 / 1000000, 1),
                 'MT, induite essentiellement par la baisse du trafic des hydrocarbures ('
                 , MCH, '). Le trafic des conteneurs a, par contre, enregistré une hausse de ', MCC, '.')
else:
    if (MCC0 > 0) & (MCH0 > 0):
        st.write('- Une progression du cabotage (', '{0:.1%}'.format(Taux_ann_Cab), '), avec un volume de'
                 , round(Tr_Glob_Cab1 / 1000000, 1),
                 'MT, induite essentiellement par la hausse du trafic des conteneurs (', MCC, ')et des hydrocarbures (',
                 MCH, ').')
    elif (MCC0 > 0) & (MCH0 < 0):
        st.write('- Une progression du cabotage (', '{0:.1%}'.format(Taux_ann_Cab), '), avec un volume de',
                 round(Tr_Glob_Cab1 / 1000000, 1),
                 'MT, induite essentiellement par la hausse du trafic des conteneurs ('
                 , MCC, '). Le trafic des hydrocarbures a, par contre, enregistré une baisse de ', abs(MCH), '.')

    elif (MCC0 < 0) & (MCH0 > 0):
        st.write('- Une progression du cabotage (', '{0:.1%}'.format(Taux_ann_Cab), '), avec un volume de',
                 round(Tr_Glob_Cab1 / 1000000, 1),
                 'MT, induite essentiellement par la hausse du trafic des hydrocarbures ('
                 , MCH, '). Le trafic des conteneurs a, par contre, enregistré une baisse de ', abs(MCC), '.')

    # In[16]:

##st.write(df.columns)


# In[17]:


# df['Année']
# df[df['Année']-1]
# df[df['Année']-1]
##st.write(len(df))

# i=1
# for i in range(int(df.shape[0])):
#   st.write(df['Valeurs'][i])
#    i=i+1


# In[18]:


##st.write(Table_Select['Marchandises'].unique())


# In[19]:


##st.write(Table_Select['Marchandises'].loc[(Table_Select.Port=="Casablanca")].unique())


# In[ ]:


# In[20]:


# Répartition par port
# Ordre ports
list_port_vol = dict()
i = 0
j = 0

for j in list(Table_Select['Port'].unique()):
    list_port_vol[j] = 0
    port_vol = 0
    for i in range(Mois_Select + 1):
        port_vol = port_vol + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Port == j) & (Table_Select.Unité == "Tonne")].sum(axis=0)
        i = i + 1
    list_port_vol[j] = port_vol
##st.write(list_port_vol)
Port_Vol = sorted(list_port_vol.items(), key=lambda t: t[1], reverse=True)
[(p1, v1), (p2, v2), (p3, v3), (p4, v4), (p5, v5), (p6, v6), (p7, v7), (p8, v8), (p9, v9), (p10, v10),
 (p11, v11)] = Port_Vol
##st.write(Port_Vol)
##st.write(len(Port_Vol))
Vol_Cas_Jorf_Moham = 0
# Représentation graphique
for i in {'Casablanca', 'Jorf Lasfar', 'Mohammedia'}:
    Vol_Cas_Jorf_Moham = Vol_Cas_Jorf_Moham + list_port_vol[i]
##st.write(Vol_Cas_Jorf_Moham)
##st.write('La répartition du trafic par port fait ressortir que les ports de Mohammedia, Casablanca et Jorf Lasfar ont assuré le transit de', round(Vol_Cas_Jorf_Moham/1000000,1),' millions de tonnes, ce qui représente environ', '{0:.1%}'.format((Vol_Cas_Jorf_Moham/Tr_Glob_i1)), 'du trafic des ports gérés par l’ANP.L’analyse des principales évolutions enregistrées par port se présente comme suit :')


# In[21]:


# st.write(dict(Port_Vol))
# st.write(dict(Port_Vol).values())
##st.write(dict(nsmallest(4, dict(Port_Vol).items(), itemgetter(1))))
# st.write(dict(Port_Vol.remove(0)))
# s=dict()
# for i,j in dict(Port_Vol).items():
#  if j!=0:
#       s[i]=j
# st.write(s)

s = dict()
s['Autres Ports'] = 0
for i, j in dict(Port_Vol).items():
    if j not in dict(nsmallest(4, dict(Port_Vol).items(), itemgetter(1))).values():
        s[i] = j
    elif j in (dict(nsmallest(4, dict(Port_Vol).items(), itemgetter(1))).values()):
        s['Autres Ports'] = s['Autres Ports'] + j
##st.write(s)

##st.write(s)


# In[22]:


# st.write(dict(Port_Vol.remove(0)))
PieChart = dict()
PieChart['Autres Ports'] = 0
for i, j in dict(Port_Vol).items():
    if j not in (dict(nsmallest(4, dict(Port_Vol).items(), itemgetter(1))).values()):
        PieChart[i] = j
    elif j in (dict(nsmallest(4, dict(Port_Vol).items(), itemgetter(1))).values()):
        PieChart['Autres Ports'] = PieChart['Autres Ports'] + j
st.write(PieChart)
# PieChart_tri=dict(sorted(PieChart.items(), key=lambda t: t[1], reverse=True))
explode = (0, 0.15, 0, 0)
#fig = plt.figure(1, figsize=(7, 7))
fig, ax = plt.subplots(figsize=(5,5))
# plt.pie(dict(Port_Vol).values(),labels=dict(Port_Vol).keys(), autopct='%1.1f%%', startangle=90, shadow=True)
a = ax.pie(PieChart.values(), labels=PieChart.keys(), autopct='%1.1f%%', startangle=90, shadow=True,
            explode=(0, 0.1, 0, 0, 0.1, 0, 0.2, 0.6))
plt.axis('equal')
st.pyplot(fig)

# In[23]:


# Trafics par port
# Analyse de (p1,v1)

list_march_p1v1 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p1)].unique()):
    list_march_p1v1[j] = 0
    march_p1 = 0
    for i in range(Mois_Select + 1):
        march_p1 = march_p1 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p1)].sum(axis=0)
        i = i + 1
    list_march_p1v1[j] = march_p1
# st.write(list_march_p1v1)


smallest3_p1 = dict(nsmallest(3, list_march_p1v1.items(), itemgetter(1)))
largest3_p1 = dict(nlargest(3, list_march_p1v1.items(), itemgetter(1)))

# st.write(smallest3_p1)
# st.write(largest3_p1)

list_march_p1v1_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p1.keys()):

    list_march_p1v1_PrNeg[j] = 0
    march_p1v1_PrNeg1 = 0
    march_p1v1_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p1v1_PrNeg1 = march_p1v1_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                           Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p1)].sum(
            axis=0)

        march_p1v1_PrNeg2 = march_p1v1_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                                 Table_Select.Marchandises == j) & (
                                                                                                 Table_Select.Unité == "Tonne") & (
                                                                                                 Table_Select.Port == p1)].sum(
            axis=0)

        i = i + 1
    march_p1v1_PrNeg = '{0:.1%}'.format((march_p1v1_PrNeg1 - march_p1v1_PrNeg2) / march_p1v1_PrNeg2)
    # st.write(march_Imp_PrNeg1)
    list_march_p1v1_PrNeg[j] = march_p1v1_PrNeg
# st.write(list_march_p1v1_PrNeg)

list_march_p1v1_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p1.keys()):

    list_march_p1v1_PrPos[j] = 0
    march_p1v1_PrPos1 = 0
    march_p1v1_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p1v1_PrPos1 = march_p1v1_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p1)].sum(axis=0)
        march_p1v1_PrPos2 = march_p1v1_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Port == p1)].sum(axis=0)

        i = i + 1
    march_p1v1_PrPos = '{0:.1%}'.format((march_p1v1_PrPos1 - march_p1v1_PrPos2) / march_p1v1_PrPos2)

    list_march_p1v1_PrPos[j] = march_p1v1_PrPos
# st.write(list_march_p1v1_PrPos)

#############

[(p1tr1, p1tx1), (p1tr2, p1tx2), (p1tr3, p1tx3)] = list_march_p1v1_PrNeg.items()
# st.write(p1tx1)

[(p1tr4, p1tx4), (p1tr5, p1tx5), (p1tr6, p1tx6)] = list_march_p1v1_PrPos.items()
# st.write(p1tr5)

#############

Trf_p1v1_Glob1 = 0
Trf_p1v1_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p1v1_Glob1 = Trf_p1v1_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p1)].sum(axis=0)
    Trf_p1v1_Glob2 = Trf_p1v1_Glob2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p1)].sum(axis=0)

    i = i + 1
Taux_Trf_p1v1_Glob = (Trf_p1v1_Glob1 - Trf_p1v1_Glob2) / Trf_p1v1_Glob2
# st.write(Trf_p1v1_Glob1, Taux_Trf_p1v1_Glob)

if Taux_Trf_p1v1_Glob < 0:
    st.write(p1, ': ', '{0:.1%}'.format(Taux_Trf_p1v1_Glob))
    st.write('Chiffré à', round(Trf_p1v1_Glob1 / 1000000, 1), ' millions de tonnes à fin', Mois2[Mois_Select - 1],
             Année_Select,
             ', le port de ', p1, 'a enregistré une quote-part de ',
             '{0:.1%}'.format(Trf_p1v1_Glob1 / Tr_Glob_i1),
             ' du trafic global. Par rapport à la même période de l’année précédente, ce port a enregistré une baisse de',
             '{0:.1%}'.format(abs(Taux_Trf_p1v1_Glob)), ', due principalement à ce qui suit:')
    st.write(' -Un recul des trafics: ', p1tr1, '(', p1tx1, '),', p1tr2, '(', p1tx2, ') et ', p1tr3, '(', p1tx3, ')')
    st.write(' -Une augmentation des trafics: ', p1tr4, '(', p1tx4, '),', p1tr5, '(', p1tx5, ') et ', p1tr6, '(', p1tx6,
             ')')

else:

    st.write(p1, ': +', '{0:.1%}'.format(Taux_Trf_p1v1_Glob))
    st.write('Chiffré à', round(Trf_p1v1_Glob1 / 1000000, 1), ' millions de tonnes à fin', Mois2[Mois_Select - 1],
             Année_Select,
             ', le port de', p1, 'a enregistré une quote-part de ',
             '{0:.1%}'.format(Trf_p1v1_Glob1 / Tr_Glob_i1),
             ' du trafic global. Par rapport à la même période de l’année précédente, ce port a enregistré une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p1v1_Glob)), ', due principalement à ce qui suit:')
    st.write(' -Une hausse des trafics: ', p1tr4, '(', p1tx4, '),', p1tr5, '(', p1tx5, ') et ', p1tr6, '(', p1tx6, ')')
    st.write(' -Une baisse des trafics: ', p1tr1, '(', p1tx1, '),', p1tr2, '(', p1tx2, ') et ', p1tr3, '(', p1tx3, ')')

# In[24]:


# Trafics par port
# Analyse de (p2,v2)

list_march_p2v2 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p2)].unique()):
    list_march_p2v2[j] = 0
    march_p2 = 0
    for i in range(Mois_Select + 1):
        march_p2 = march_p2 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p2)].sum(axis=0)
        i = i + 1
    list_march_p2v2[j] = march_p2
# st.write(list_march_p2v2)


smallest3_p2 = dict(nsmallest(3, list_march_p2v2.items(), itemgetter(1)))
largest3_p2 = dict(nlargest(3, list_march_p2v2.items(), itemgetter(1)))

# st.write(smallest3_p2)
# st.write(largest3_p2)

list_march_p2v2_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p2.keys()):

    list_march_p2v2_PrNeg[j] = 0
    march_p2v2_PrNeg1 = 0
    march_p2v2_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p2v2_PrNeg1 = march_p2v2_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                           Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p2)].sum(
            axis=0)

        march_p2v2_PrNeg2 = march_p2v2_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                                 Table_Select.Marchandises == j) & (
                                                                                             Table_Select.Unité == "Tonne") & (
                                                                                             Table_Select.Port == p2)].sum(
            axis=0)

        i = i + 1
    march_p2v2_PrNeg = '{0:.1%}'.format((march_p2v2_PrNeg1 - march_p2v2_PrNeg2) / march_p2v2_PrNeg2)

    list_march_p2v2_PrNeg[j] = march_p2v2_PrNeg
# st.write(list_march_p2v2_PrNeg)

list_march_p2v2_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p2.keys()):

    list_march_p2v2_PrPos[j] = 0
    march_p2v2_PrPos1 = 0
    march_p2v2_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p2v2_PrPos1 = march_p2v2_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p2)].sum(axis=0)
        march_p2v2_PrPos2 = march_p2v2_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p2)].sum(axis=0)

        i = i + 1
    march_p2v2_PrPos = '{0:.1%}'.format((march_p2v2_PrPos1 - march_p2v2_PrPos2) / march_p2v2_PrPos2)

    list_march_p2v2_PrPos[j] = march_p2v2_PrPos
# st.write(list_march_p2v2_PrPos)

#############

[(p2tr1, p2tx1), (p2tr2, p2tx2), (p2tr3, p2tx3)] = list_march_p2v2_PrNeg.items()
# st.write(p2tx1)

[(p2tr4, p2tx4), (p2tr5, p2tx5), (p2tr6, p2tx6)] = list_march_p2v2_PrPos.items()
# st.write(p2tr5)

#############

Trf_p2v2_Glob1 = 0
Trf_p2v2_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p2v2_Glob1 = Trf_p2v2_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") &
        (df.Port == p2)].sum(axis=0)
    Trf_p2v2_Glob2 = Trf_p2v2_Glob2 + df['Valeurs'].loc[(df.Année == (Année_Select - 1)) &
                                                        (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p2)].sum(
        axis=0)

    i = i + 1
Taux_Trf_p2v2_Glob = (Trf_p2v2_Glob1 - Trf_p2v2_Glob2) / Trf_p2v2_Glob2
# st.write(Trf_p2v2_Glob1, Taux_Trf_p2v2_Glob)

if Taux_Trf_p2v2_Glob < 0:
    st.write(p2, ': ', '{0:.1%}'.format(Taux_Trf_p2v2_Glob))
    st.write('Chiffré à', round(Trf_p2v2_Glob1 / 1000000, 1), ' millions de tonnes à fin', Mois2[Mois_Select - 1],
             Année_Select,
             ', le port de', p2, 'a enregistré une quote-part de ',
             '{0:.1%}'.format(Trf_p2v2_Glob1 / Tr_Glob_i1),
             ' du trafic global. Par rapport à la même période de l’année précédente, ce port a enregistré un recul de',
             '{0:.1%}'.format(abs(Taux_Trf_p2v2_Glob)), ', due principalement à ce qui suit:')
    st.write(' -Une baisse des trafics: ', p2tr1, '(', p2tx1, '),', p2tr2, '(', p2tx2, ') et ', p2tr3, '(', p2tx3, ');')
    st.write(' -Une hausse des trafics: ', p2tr4, '(', p2tx4, '),', p2tr5, '(', p2tx5, ') et ', p2tr6, '(', p2tx6, ').')

else:

    st.write(p2, ': +', '{0:.1%}'.format(Taux_Trf_p2v2_Glob))
    st.write('Chiffré à', round(Trf_p2v2_Glob1 / 1000000, 1), ' millions de tonnes à fin', Mois2[Mois_Select - 1],
             Année_Select,
             ', le port de', p2, 'a enregistré une quote-part de ',
             '{0:.1%}'.format(Trf_p2v2_Glob1 / Tr_Glob_i1),
             ' du trafic global. Par rapport à la même période de l’année précédente, ce port a enregistré une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p2v2_Glob)), ', due principalement à ce qui suit:')
    st.write(' -Une hausse des trafics: ', p2tr4, '(', p2tx4, '),', p2tr5, '(', p2tx5, ') et ', p2tr6, '(', p2tx6, ')')
    st.write(' -Une baisse des trafics: ', p2tr1, '(', p2tx1, '),', p2tr2, '(', p2tx2, ') et ', p2tr3, '(', p2tx3, ').')

# In[25]:


# Trafics par port
# Analyse de (p3,v3)

list_march_p3v3 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p3)].unique()):
    list_march_p3v3[j] = 0
    march_p3 = 0
    for i in range(Mois_Select + 1):
        march_p3 = march_p3 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p3)].sum(axis=0)
        i = i + 1
    list_march_p3v3[j] = march_p3
# st.write(list_march_p3v3)


smallest3_p3 = dict(nsmallest(3, list_march_p3v3.items(), itemgetter(1)))
largest3_p3 = dict(nlargest(3, list_march_p3v3.items(), itemgetter(1)))

# st.write(smallest3_p3)
# st.write(largest3_p3)

list_march_p3v3_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p3.keys()):

    list_march_p3v3_PrNeg[j] = 0
    march_p3v3_PrNeg1 = 0
    march_p3v3_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p3v3_PrNeg1 = march_p3v3_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                           Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p3)].sum(
            axis=0)

        march_p3v3_PrNeg2 = march_p3v3_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                             Table_Select.Marchandises == j) & (
                                                                                             Table_Select.Unité == "Tonne") & (
                                                                                                 Table_Select.Port == p3)].sum(
            axis=0)

        i = i + 1
    march_p3v3_PrNeg = '{0:.1%}'.format((march_p3v3_PrNeg1 - march_p3v3_PrNeg2) / march_p3v3_PrNeg2)

    list_march_p3v3_PrNeg[j] = march_p3v3_PrNeg
# st.write(list_march_p3v3_PrNeg)

list_march_p3v3_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p3.keys()):

    list_march_p3v3_PrPos[j] = 0
    march_p3v3_PrPos1 = 0
    march_p3v3_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p3v3_PrPos1 = march_p3v3_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Port == p3)].sum(axis=0)
        march_p3v3_PrPos2 = march_p3v3_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p3)].sum(axis=0)

        i = i + 1
    march_p3v3_PrPos = '{0:.1%}'.format((march_p3v3_PrPos1 - march_p3v3_PrPos2) / march_p3v3_PrPos2)

    list_march_p3v3_PrPos[j] = march_p3v3_PrPos
# st.write(list_march_p3v3_PrPos)

#############

[(p3tr1, p3tx1), (p3tr2, p3tx2), (p3tr3, p3tx3)] = list_march_p3v3_PrNeg.items()
# st.write(p3tx1)

[(p3tr4, p3tx4), (p3tr5, p3tx5), (p3tr6, p3tx6)] = list_march_p3v3_PrPos.items()
# st.write(p3tr5)

#############

Trf_p3v3_Glob1 = 0
Trf_p3v3_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p3v3_Glob1 = Trf_p3v3_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") &
        (df.Port == p3)].sum(axis=0)
    Trf_p3v3_Glob2 = Trf_p3v3_Glob2 + df['Valeurs'].loc[(df.Année == (Année_Select - 1)) &
                                                        (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p3)].sum(
        axis=0)

    i = i + 1
Taux_Trf_p3v3_Glob = (Trf_p3v3_Glob1 - Trf_p3v3_Glob2) / Trf_p3v3_Glob2
# st.write(Trf_p3v3_Glob1, Taux_Trf_p3v3_Glob)

if Taux_Trf_p3v3_Glob < 0:
    st.write(p3, ': ', '{0:.1%}'.format(Taux_Trf_p3v3_Glob))
    st.write('Avec un volume de ', round(Trf_p3v3_Glob1 / 1000000, 1),
             ' millions de tonnes à fin', Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une régression de',
             '{0:.1%}'.format(abs(Taux_Trf_p3v3_Glob)),
             ', due principalement à ce qui suit:')
    st.write(' -Une baisse des trafics: ', p3tr1, '(', p3tx1, '),', p3tr2, '(', p3tx2, ') et ', p3tr3, '(', p3tx3, ');')
    st.write(' -Une hausse des trafics: ', p3tr4, '(', p3tx4, '),', p3tr5, '(', p3tx5, ') et ', p3tr6, '(', p3tx6, ').')

else:

    st.write(p3, ': +', '{0:.1%}'.format(Taux_Trf_p3v3_Glob))
    st.write('Avec un volume de ', round(Trf_p3v3_Glob1 / 1000000, 1), ' millions de tonnes à fin',
             Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p3v3_Glob)), ', due principalement à ce qui suit:')
    st.write(' -Une hausse des trafics: ', p3tr4, '(', p3tx4, '),', p3tr5, '(', p3tx5, ') et ', p3tr6, '(', p3tx6, ')')
    st.write(' -Une baisse des trafics: ', p3tr1, '(', p3tx1, '),', p3tr2, '(', p3tx2, ') et ', p3tr3, '(', p3tx3, ').')

# In[26]:


list_march_p4v4 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p4)].unique()):
    list_march_p4v4[j] = 0
    march_p4 = 0
    for i in range(Mois_Select + 1):
        march_p4 = march_p4 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p4)].sum(axis=0)
        i = i + 1
    list_march_p4v4[j] = march_p4
# st.write(list_march_p4v4)


smallest3_p4 = dict(nsmallest(3, list_march_p4v4.items(), itemgetter(1)))
largest3_p4 = dict(nlargest(3, list_march_p4v4.items(), itemgetter(1)))

# st.write(smallest3_p4)
# st.write(largest3_p4)

list_march_p4v4_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p4.keys()):

    list_march_p4v4_PrNeg[j] = 0
    march_p4v4_PrNeg1 = 0
    march_p4v4_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p4v4_PrNeg1 = march_p4v4_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                           Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p4)].sum(
            axis=0)

        march_p4v4_PrNeg2 = march_p4v4_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                             Table_Select.Marchandises == j) & (
                                                                                             Table_Select.Unité == "Tonne") & (
                                                                                             Table_Select.Port == p4)].sum(
            axis=0)

        i = i + 1
    march_p4v4_PrNeg = '{0:.1%}'.format((march_p4v4_PrNeg1 - march_p4v4_PrNeg2) / march_p4v4_PrNeg2)

    list_march_p4v4_PrNeg[j] = march_p4v4_PrNeg
# st.write(list_march_p4v4_PrNeg)

list_march_p4v4_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p4.keys()):

    list_march_p4v4_PrPos[j] = 0
    march_p4v4_PrPos1 = 0
    march_p4v4_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p4v4_PrPos1 = march_p4v4_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p4)].sum(axis=0)
        march_p4v4_PrPos2 = march_p4v4_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p4)].sum(axis=0)

        i = i + 1
    march_p4v4_PrPos = '{0:.1%}'.format((march_p4v4_PrPos1 - march_p4v4_PrPos2) / march_p4v4_PrPos2)

    list_march_p4v4_PrPos[j] = march_p4v4_PrPos
# st.write(list_march_p4v4_PrPos)

#############

[(p4tr1, p4tx1), (p4tr2, p4tx2), (p4tr3, p4tx3)] = list_march_p4v4_PrNeg.items()
# st.write(p4tx1)

[(p4tr4, p4tx4), (p4tr5, p4tx5), (p4tr6, p4tx6)] = list_march_p4v4_PrPos.items()
# st.write(p4tr5)

#############

Trf_p4v4_Glob1 = 0
Trf_p4v4_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p4v4_Glob1 = Trf_p4v4_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") &
        (df.Port == p4)].sum(axis=0)
    Trf_p4v4_Glob2 = Trf_p4v4_Glob2 + df['Valeurs'].loc[(df.Année == (Année_Select - 1)) &
                                                        (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p4)].sum(
        axis=0)

    i = i + 1
Taux_Trf_p4v4_Glob = (Trf_p4v4_Glob1 - Trf_p4v4_Glob2) / Trf_p4v4_Glob2
# st.write(Trf_p4v4_Glob1, Taux_Trf_p4v4_Glob)

if Taux_Trf_p4v4_Glob < 0:
    st.write(p4, ': ', '{0:.1%}'.format(Taux_Trf_p4v4_Glob))
    st.write('Avec un volume de ', round(Trf_p4v4_Glob1 / 1000000, 1),
             ' millions de tonnes à fin', Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une régression de',
             '{0:.1%}'.format(abs(Taux_Trf_p4v4_Glob)),
             ', due principalement à ce qui suit:')
    st.write(' -Une baisse des trafics: ', p4tr1, '(', p4tx1, '),', p4tr2, '(', p4tx2, ') et ', p4tr3, '(', p4tx3, ');')
    st.write(' -Une hausse des trafics: ', p4tr4, '(', p4tx4, '),', p4tr5, '(', p4tx5, ') et ', p4tr6, '(', p4tx6, ').')

else:

    st.write(p4, ': +', '{0:.1%}'.format(Taux_Trf_p4v4_Glob))
    st.write('Avec un volume de ', round(Trf_p4v4_Glob1 / 1000000, 1), ' millions de tonnes à fin',
             Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p4v4_Glob)), ', due principalement à ce qui suit:')
    st.write(' -Une hausse des trafics: ', p4tr4, '(', p4tx4, '),', p4tr5, '(', p4tx5, ') et ', p4tr6, '(', p4tx6, ')')
    st.write(' -Une baisse des trafics: ', p4tr1, '(', p4tx1, '),', p4tr2, '(', p4tx2, ') et ', p4tr3, '(', p4tx3, ').')

# In[27]:


# Trafics par port
# Analyse de (p5,v5)

list_march_p5v5 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p5)].unique()):
    list_march_p5v5[j] = 0
    march_p5 = 0
    for i in range(Mois_Select + 1):
        march_p5 = march_p5 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p5)].sum(axis=0)
        i = i + 1
    list_march_p5v5[j] = march_p5
# st.write(list_march_p5v5)


smallest3_p5 = dict(nsmallest(3, list_march_p5v5.items(), itemgetter(1)))
largest3_p5 = dict(nlargest(3, list_march_p5v5.items(), itemgetter(1)))

# st.write(smallest3_p5)
# st.write(largest3_p5)

list_march_p5v5_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p5.keys()):

    list_march_p5v5_PrNeg[j] = 0
    march_p5v5_PrNeg1 = 0
    march_p5v5_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p5v5_PrNeg1 = march_p5v5_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                           Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p5)].sum(
            axis=0)

        march_p5v5_PrNeg2 = march_p5v5_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                                 Table_Select.Marchandises == j) & (
                                                                                                 Table_Select.Unité == "Tonne") & (
                                                                                                 Table_Select.Port == p5)].sum(
            axis=0)

        i = i + 1
    march_p5v5_PrNeg = '{0:.1%}'.format((march_p5v5_PrNeg1 - march_p5v5_PrNeg2) / march_p5v5_PrNeg2)

    list_march_p5v5_PrNeg[j] = march_p5v5_PrNeg
# st.write(list_march_p5v5_PrNeg)

list_march_p5v5_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p5.keys()):

    list_march_p5v5_PrPos[j] = 0
    march_p5v5_PrPos1 = 0
    march_p5v5_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p5v5_PrPos1 = march_p5v5_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p5)].sum(axis=0)
        march_p5v5_PrPos2 = march_p5v5_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p5)].sum(axis=0)

        i = i + 1
    march_p5v5_PrPos = '{0:.1%}'.format((march_p5v5_PrPos1 - march_p5v5_PrPos2) / march_p5v5_PrPos2)

    list_march_p5v5_PrPos[j] = march_p5v5_PrPos
# st.write(list_march_p5v5_PrPos)

#############

[(p5tr1, p5tx1), (p5tr2, p5tx2), (p5tr3, p5tx3)] = list_march_p5v5_PrNeg.items()
# st.write(p5tx1)

[(p5tr4, p5tx4), (p5tr5, p5tx5), (p5tr6, p5tx6)] = list_march_p5v5_PrPos.items()
# st.write(p5tr5)

#############

Trf_p5v5_Glob1 = 0
Trf_p5v5_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p5v5_Glob1 = Trf_p5v5_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p5)].sum(axis=0)
    Trf_p5v5_Glob2 = Trf_p5v5_Glob2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p5)].sum(axis=0)

    i = i + 1

Taux_Trf_p5v5_Glob = (Trf_p5v5_Glob1 - Trf_p5v5_Glob2) / Trf_p5v5_Glob2
# st.write(Trf_p5v5_Glob1, Taux_Trf_p5v5_Glob)

if Taux_Trf_p5v5_Glob < 0:
    st.write(p5, ': ', '{0:.1%}'.format(Taux_Trf_p5v5_Glob))
    st.write('Avec un volume de ', round(Trf_p5v5_Glob1 / 1000000, 1), ' millions de tonnes à fin',
             Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une régression de',
             '{0:.1%}'.format(abs(Taux_Trf_p5v5_Glob)),
             ', due principalement à ce qui suit:')
    st.write(' -Une baisse des trafics: ', p5tr1, '(', p5tx1, '),', p5tr2, '(', p5tx2, ') et ', p5tr3, '(', p5tx3, ');')
    st.write(' -Une hausse des trafics: ', p5tr4, '(', p5tx4, '),', p5tr5, '(', p5tx5, ') et ', p5tr6, '(', p5tx6, ').')

else:

    st.write(p5, ': +', '{0:.1%}'.format(Taux_Trf_p5v5_Glob))
    st.write('Avec un volume de ', round(Trf_p5v5_Glob1 / 1000000, 1), ' millions de tonnes à fin',
             Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p5v5_Glob)), ', due principalement à ce qui suit:')
    st.write(' -Une hausse des trafics: ', p5tr4, '(', p5tx4, '),', p5tr5, '(', p5tx5, ') et ', p5tr6, '(', p5tx6, ')')
    st.write(' -Une baisse des trafics: ', p5tr1, '(', p5tx1, '),', p5tr2, '(', p5tx2, ') et ', p5tr3, '(', p5tx3, ').')

# In[28]:


# Trafics par port
# Analyse de (p6,v6)

list_march_p6v6 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p6)].unique()):
    list_march_p6v6[j] = 0
    march_p6 = 0
    for i in range(Mois_Select + 1):
        march_p6 = march_p6 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p6)].sum(axis=0)
        i = i + 1
    list_march_p6v6[j] = march_p6
# st.write(list_march_p6v6)


smallest3_p6 = dict(nsmallest(3, list_march_p6v6.items(), itemgetter(1)))
largest3_p6 = dict(nlargest(3, list_march_p6v6.items(), itemgetter(1)))

# st.write(smallest3_p6)
# st.write(largest3_p6)

list_march_p6v6_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p6.keys()):

    list_march_p6v6_PrNeg[j] = 0
    march_p6v6_PrNeg1 = 0
    march_p6v6_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p6v6_PrNeg1 = march_p6v6_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                           Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p6)].sum(
            axis=0)

        march_p6v6_PrNeg2 = march_p6v6_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                             Table_Select.Marchandises == j) & (
                                                                                                 Table_Select.Unité == "Tonne") & (
                                                                                                 Table_Select.Port == p6)].sum(
            axis=0)

        i = i + 1
    march_p6v6_PrNeg = '{0:.1%}'.format((march_p6v6_PrNeg1 - march_p6v6_PrNeg2) / march_p6v6_PrNeg2)

    list_march_p6v6_PrNeg[j] = march_p6v6_PrNeg
# st.write(list_march_p6v6_PrNeg)

list_march_p6v6_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p6.keys()):

    list_march_p6v6_PrPos[j] = 0
    march_p6v6_PrPos1 = 0
    march_p6v6_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p6v6_PrPos1 = march_p6v6_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p6)].sum(axis=0)
        march_p6v6_PrPos2 = march_p6v6_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p6)].sum(axis=0)

        i = i + 1
    march_p6v6_PrPos = '{0:.1%}'.format((march_p6v6_PrPos1 - march_p6v6_PrPos2) / march_p6v6_PrPos2)

    list_march_p6v6_PrPos[j] = march_p6v6_PrPos
# st.write(list_march_p6v6_PrPos)

#############

[(p6tr1, p6tx1), (p6tr2, p6tx2), (p6tr3, p6tx3)] = list_march_p6v6_PrNeg.items()
# st.write(p6tx1)

[(p6tr4, p6tx4), (p6tr5, p6tx5), (p6tr6, p6tx6)] = list_march_p6v6_PrPos.items()
# st.write(p6tr5)

#############

Trf_p6v6_Glob1 = 0
Trf_p6v6_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p6v6_Glob1 = Trf_p6v6_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") &
        (df.Port == p6)].sum(axis=0)
    Trf_p6v6_Glob2 = Trf_p6v6_Glob2 + df['Valeurs'].loc[(df.Année == (Année_Select - 1)) &
                                                        (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p6)].sum(
        axis=0)

    i = i + 1
Taux_Trf_p6v6_Glob = (Trf_p6v6_Glob1 - Trf_p6v6_Glob2) / Trf_p6v6_Glob2
# st.write(Trf_p6v6_Glob1, Taux_Trf_p6v6_Glob)

if Taux_Trf_p6v6_Glob < 0:
    st.write(p6, ': ', '{0:.1%}'.format(Taux_Trf_p6v6_Glob))
    st.write('Avec un volume de ', round(Trf_p6v6_Glob1 / 1000000, 2),
             ' millions de tonnes à fin', Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une régression de',
             '{0:.1%}'.format(abs(Taux_Trf_p6v6_Glob)),
             ', due principalement à ce qui suit:')
    st.write(' -Une baisse des trafics: ', p6tr1, '(', p6tx1, '),', p6tr2, '(', p6tx2, ') et ', p6tr3, '(', p6tx3, ');')
    st.write(' -Une hausse des trafics: ', p6tr4, '(', p6tx4, '),', p6tr5, '(', p6tx5, ') et ', p6tr6, '(', p6tx6, ').')

else:

    st.write(p6, ': +', '{0:.1%}'.format(Taux_Trf_p6v6_Glob))
    st.write('Avec un volume de ', round(Trf_p6v6_Glob1 / 1000000, 2), ' millions de tonnes à fin',
             Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p6v6_Glob)), ', due principalement à ce qui suit:')
    st.write(' -Une hausse des trafics: ', p6tr4, '(', p6tx4, '),', p6tr5, '(', p6tx5, ') et ', p6tr6, '(', p6tx6, ')')
    st.write(' -Une baisse des trafics: ', p6tr1, '(', p6tx1, '),', p6tr2, '(', p6tx2, ') et ', p6tr3, '(', p6tx3, ').')

# In[29]:


# Trafics par port
# Analyse de (p7,v7)

list_march_p7v7 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p7)].unique()):
    list_march_p7v7[j] = 0
    march_p7 = 0
    for i in range(Mois_Select + 1):
        march_p7 = march_p7 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p7)].sum(axis=0)
        i = i + 1
    list_march_p7v7[j] = march_p7
# st.write(list_march_p7v7)


smallest3_p7 = dict(nsmallest(3, list_march_p7v7.items(), itemgetter(1)))
largest3_p7 = dict(nlargest(3, list_march_p7v7.items(), itemgetter(1)))

# st.write(smallest3_p7)
# st.write(largest3_p7)

list_march_p7v7_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p7.keys()):

    list_march_p7v7_PrNeg[j] = 0
    march_p7v7_PrNeg1 = 0
    march_p7v7_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p7v7_PrNeg1 = march_p7v7_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                       Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p7)].sum(
            axis=0)

        march_p7v7_PrNeg2 = march_p7v7_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                                 Table_Select.Marchandises == j) & (
                                                                                                 Table_Select.Unité == "Tonne") & (
                                                                                                 Table_Select.Port == p7)].sum(
            axis=0)

        i = i + 1
    march_p7v7_PrNeg = '{0:.1%}'.format((march_p7v7_PrNeg1 - march_p7v7_PrNeg2) / march_p7v7_PrNeg2)

    list_march_p7v7_PrNeg[j] = march_p7v7_PrNeg
# st.write(list_march_p7v7_PrNeg)

list_march_p7v7_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p7.keys()):

    list_march_p7v7_PrPos[j] = 0
    march_p7v7_PrPos1 = 0
    march_p7v7_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p7v7_PrPos1 = march_p7v7_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p7)].sum(axis=0)
        march_p7v7_PrPos2 = march_p7v7_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Port == p7)].sum(axis=0)

        i = i + 1
    march_p7v7_PrPos = '{0:.1%}'.format((march_p7v7_PrPos1 - march_p7v7_PrPos2) / march_p7v7_PrPos2)

    list_march_p7v7_PrPos[j] = march_p7v7_PrPos
# st.write(list_march_p7v7_PrPos)
# st.write(list_march_p7v7_PrPos)
# st.write(p7)
#############


[(p7tr1, p7tx1), (p7tr2, p7tx2), (p7tr3, p7tx3)] = list_march_p7v7_PrNeg.items()
# st.write(p7tx1)

[(p7tr4, p7tx4), (p7tr5, p7tx5), (p7tr6, p7tx6)] = list_march_p7v7_PrPos.items()
# st.write(p7tr5)

#############

Trf_p7v7_Glob1 = 0
Trf_p7v7_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p7v7_Glob1 = Trf_p7v7_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") &
        (df.Port == p7)].sum(axis=0)
    Trf_p7v7_Glob2 = Trf_p7v7_Glob2 + df['Valeurs'].loc[(df.Année == (Année_Select - 1)) &
                                                        (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p7)].sum(
        axis=0)

    i = i + 1
Taux_Trf_p7v7_Glob = (Trf_p7v7_Glob1 - Trf_p7v7_Glob2) / Trf_p7v7_Glob2
# st.write(Trf_p7v7_Glob1, Taux_Trf_p7v7_Glob)

if Taux_Trf_p7v7_Glob < 0:
    st.write(p7, ': ', '{0:.1%}'.format(Taux_Trf_p7v7_Glob))
    st.write('Avec un volume de ', '{:,}'.format(int(Trf_p7v7_Glob1)).replace(',', ' '),
             ' tonnes à fin', Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une régression de',
             '{0:.1%}'.format(abs(Taux_Trf_p7v7_Glob)),
             ', due principalement à la baisse des trafics: ', p7tr1, '(', p7tx1, ') et', p7tr2, '(', p7tx2, ').')

else:
    st.write(p7, ': +', '{0:.1%}'.format(Taux_Trf_p7v7_Glob))
    st.write('Avec un volume de ', '{:,}'.format(int(Trf_p7v7_Glob1)).replace(',', ' '), ' tonnes à fin',
             Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p7v7_Glob)), ', due principalement à la hausse des trafics: ', p7tr4, '(',
             p7tx4, ') et', p7tr5, '(', p7tx5, ').')

# In[30]:


# Trafics par port
# Analyse de (p8,v8)

list_march_p8v8 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p8)].unique()):
    list_march_p8v8[j] = 0
    march_p8 = 0
    for i in range(Mois_Select + 1):
        march_p8 = march_p8 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p8)].sum(axis=0)
        i = i + 1
    list_march_p8v8[j] = march_p8
# st.write(list_march_p8v8)


smallest3_p8 = dict(nsmallest(3, list_march_p8v8.items(), itemgetter(1)))
largest3_p8 = dict(nlargest(3, list_march_p8v8.items(), itemgetter(1)))

# st.write(smallest3_p8)
# st.write(largest3_p8)

list_march_p8v8_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p8.keys()):

    list_march_p8v8_PrNeg[j] = 0
    march_p8v8_PrNeg1 = 0
    march_p8v8_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p8v8_PrNeg1 = march_p8v8_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                       Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p8)].sum(
            axis=0)

        march_p8v8_PrNeg2 = march_p8v8_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                                 Table_Select.Marchandises == j) & (
                                                                                                 Table_Select.Unité == "Tonne") & (
                                                                                             Table_Select.Port == p8)].sum(
            axis=0)

        i = i + 1
    march_p8v8_PrNeg = '{0:.1%}'.format((march_p8v8_PrNeg1 - march_p8v8_PrNeg2) / march_p8v8_PrNeg2)

    list_march_p8v8_PrNeg[j] = march_p8v8_PrNeg
# st.write(list_march_p8v8_PrNeg)

list_march_p8v8_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p8.keys()):

    list_march_p8v8_PrPos[j] = 0
    march_p8v8_PrPos1 = 0
    march_p8v8_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p8v8_PrPos1 = march_p8v8_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p8)].sum(axis=0)
        march_p8v8_PrPos2 = march_p8v8_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                        Table_Select.Port == p8)].sum(axis=0)

        i = i + 1
    march_p8v8_PrPos = '{0:.1%}'.format((march_p8v8_PrPos1 - march_p8v8_PrPos2) / march_p8v8_PrPos2)

    list_march_p8v8_PrPos[j] = march_p8v8_PrPos
# st.write(list_march_p8v8_PrPos)
# st.write(list_march_p8v8_PrPos)
# st.write(p8)
#############


[(p8tr1, p8tx1), (p8tr2, p8tx2), (p8tr3, p8tx3)] = list_march_p8v8_PrNeg.items()
# st.write(p8tx1)

[(p8tr4, p8tx4), (p8tr5, p8tx5), (p8tr6, p8tx6)] = list_march_p8v8_PrPos.items()
# st.write(p8tr5)

#############

Trf_p8v8_Glob1 = 0
Trf_p8v8_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p8v8_Glob1 = Trf_p8v8_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") &
        (df.Port == p8)].sum(axis=0)
    Trf_p8v8_Glob2 = Trf_p8v8_Glob2 + df['Valeurs'].loc[(df.Année == (Année_Select - 1)) &
                                                        (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p8)].sum(
        axis=0)

    i = i + 1
Taux_Trf_p8v8_Glob = (Trf_p8v8_Glob1 - Trf_p8v8_Glob2) / Trf_p8v8_Glob2
# st.write(Trf_p8v8_Glob1, Taux_Trf_p8v8_Glob)

if Taux_Trf_p8v8_Glob < 0:
    st.write(p8, ': ', '{0:.1%}'.format(Taux_Trf_p8v8_Glob))
    st.write('Avec un volume de ', '{:,}'.format(int(Trf_p8v8_Glob1)).replace(',', ' '),
             ' tonnes à fin', Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une régression de',
             '{0:.1%}'.format(abs(Taux_Trf_p8v8_Glob)),
             ', due principalement à la baisse des trafics: ', p8tr1, '(', p8tx1, ') et ', p8tr2, '(', p8tx2, ').')

else:

    st.write(p8, ': +', '{0:.1%}'.format(Taux_Trf_p8v8_Glob))
    st.write('Avec un volume de ', '{:,}'.format(int(Trf_p8v8_Glob1)).replace(',', ' '), ' tonnes à fin',
             Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p8v8_Glob)), ', due principalement à la hausse des trafics: ', p8tr4, '(',
             p8tx4, ') et ', p8tr5, '(', p8tx5, ').')

# In[31]:


# Trafics par port
# Analyse de (p9,v9)

list_march_p9v9 = dict()
i = 0
j = 0
for j in list(Table_Select['Marchandises'].loc[(Table_Select.Port == p9)].unique()):
    list_march_p9v9[j] = 0
    march_p9 = 0
    for i in range(Mois_Select + 1):
        march_p9 = march_p9 + Table_Select['Value_Diff'].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Port == p9)].sum(axis=0)
        i = i + 1
    list_march_p9v9[j] = march_p9
# st.write(list_march_p9v9)


smallest3_p9 = dict(nsmallest(3, list_march_p9v9.items(), itemgetter(1)))
largest3_p9 = dict(nlargest(3, list_march_p9v9.items(), itemgetter(1)))

# st.write(smallest3_p9)
# st.write(largest3_p9)

list_march_p9v9_PrNeg = dict()
i = 0
j = 0

for j in list(smallest3_p9.keys()):

    list_march_p9v9_PrNeg[j] = 0
    march_p9v9_PrNeg1 = 0
    march_p9v9_PrNeg2 = 0

    for i in range(Mois_Select + 1):
        march_p9v9_PrNeg1 = march_p9v9_PrNeg1 + Table_Select[Année_Select].loc[(Table_Select.Mois == i) &
                                                                               (Table_Select.Marchandises == j) & (
                                                                                           Table_Select.Unité == "Tonne") & (
                                                                                           Table_Select.Port == p9)].sum(
            axis=0)

        march_p9v9_PrNeg2 = march_p9v9_PrNeg2 + Table_Select[(Année_Select - 1)].loc[(Table_Select.Mois == i) &
                                                                                     (
                                                                                                 Table_Select.Marchandises == j) & (
                                                                                                 Table_Select.Unité == "Tonne") & (
                                                                                                 Table_Select.Port == p9)].sum(
            axis=0)

        i = i + 1
    march_p9v9_PrNeg = '{0:.1%}'.format((march_p9v9_PrNeg1 - march_p9v9_PrNeg2) / march_p9v9_PrNeg2)

    list_march_p9v9_PrNeg[j] = march_p9v9_PrNeg
# st.write(list_march_p9v9_PrNeg)

list_march_p9v9_PrPos = dict()
i = 0
j = 0

for j in list(largest3_p9.keys()):

    list_march_p9v9_PrPos[j] = 0
    march_p9v9_PrPos1 = 0
    march_p9v9_PrPos2 = 0

    for i in range(Mois_Select + 1):
        march_p9v9_PrPos1 = march_p9v9_PrPos1 + Table_Select[Année_Select].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Port == p9)].sum(axis=0)
        march_p9v9_PrPos2 = march_p9v9_PrPos2 + Table_Select[(Année_Select - 1)].loc[
            (Table_Select.Mois == i) & (Table_Select.Marchandises == j) & (Table_Select.Unité == "Tonne") & (
                    Table_Select.Port == p9)].sum(axis=0)

        i = i + 1
    march_p9v9_PrPos = '{0:.1%}'.format((march_p9v9_PrPos1 - march_p9v9_PrPos2) / march_p9v9_PrPos2)

    list_march_p9v9_PrPos[j] = march_p9v9_PrPos
# st.write(list_march_p9v9_PrPos)
# st.write(list_march_p9v9_PrPos)
# st.write(p9)
#############


[(p9tr1, p9tx1), (p9tr2, p9tx2), (p9tr3, p9tx3)] = list_march_p9v9_PrNeg.items()
# st.write(p9tx1)

[(p9tr4, p9tx4), (p9tr5, p9tx5), (p9tr6, p9tx6)] = list_march_p9v9_PrPos.items()
# st.write(p9tr5)

#############

Trf_p9v9_Glob1 = 0
Trf_p9v9_Glob2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_p9v9_Glob1 = Trf_p9v9_Glob1 + df['Valeurs'].loc[
        (df.Année == Année_Select) & (df.Mois == i) & (df.Unité == "Tonne") &
        (df.Port == p9)].sum(axis=0)
    Trf_p9v9_Glob2 = Trf_p9v9_Glob2 + df['Valeurs'].loc[(df.Année == (Année_Select - 1)) &
                                                        (df.Mois == i) & (df.Unité == "Tonne") & (df.Port == p9)].sum(
        axis=0)

    i = i + 1
Taux_Trf_p9v9_Glob = (Trf_p9v9_Glob1 - Trf_p9v9_Glob2) / Trf_p9v9_Glob2
# st.write(Trf_p9v9_Glob1, Taux_Trf_p9v9_Glob)

if Taux_Trf_p9v9_Glob < 0:
    st.write(p9, ': ', '{0:.1%}'.format(Taux_Trf_p9v9_Glob))
    st.write('Avec un volume de ', '{:,}'.format(int(Trf_p9v9_Glob1)).replace(',', ' '),
             '  tonnes à fin', Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une régression de',
             '{0:.1%}'.format(abs(Taux_Trf_p9v9_Glob)),
             ', due principalement à la baisse des trafics: ', p9tr1, '(', p9tx1, ') et ', p9tr2, '(', p9tx2, ').')

else:

    st.write(p9, ': +', '{0:.1%}'.format(Taux_Trf_p9v9_Glob))
    st.write('Avec un volume de ', '{:,}'.format(int(Trf_p9v9_Glob1)).replace(',', ' '), ' tonnes à fin',
             Mois2[Mois_Select - 1], Année_Select,
             ', l’activité de ce port a marqué une augmentation de',
             '{0:.1%}'.format(abs(Taux_Trf_p9v9_Glob)), ', due principalement à la hausse des trafics: ', p9tr4, '('
             , p9tx4, ') et ', p9tr5, '(', p9tx5, ').')

# In[32]:


# '{:,}'.format(1234567890.001).replace(',',' ')


# In[33]:


# présentation Trafic conteneurs
Trf_cont_EVP1 = 0
Trf_cont_EVP2 = 0
Trf_cont_EVP_Imp1 = 0
Trf_cont_EVP_Imp2 = 0
Trf_cont_EVP_Exp1 = 0
Trf_cont_EVP_Exp2 = 0
Trf_cont_EVP_Cab1 = 0
Trf_cont_EVP_Cab2 = 0
Trf_cont_EVP_Exp_vid1 = 0
Trf_cont_EVP_Exp_vid2 = 0

i = 0
for i in range(Mois_Select + 1):
    Trf_cont_EVP1 = Trf_cont_EVP1 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
        2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Marchandises == "Conteneurs")].sum(
        axis=0)
    Trf_cont_EVP2 = Trf_cont_EVP2 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
        2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Nombre") & (
                    df.Marchandises == "Conteneurs")].sum(axis=0)
    # Import
    Trf_cont_EVP_Imp1 = Trf_cont_EVP_Imp1 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
        2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Marchandises == "Conteneurs") & (
                    df.Sens_ImpExpCab == "Import")].sum(axis=0)
    Trf_cont_EVP_Imp2 = Trf_cont_EVP_Imp2 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
        2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Nombre") & (
                df.Marchandises == "Conteneurs") & (df.Sens_ImpExpCab == "Import")].sum(axis=0)
    # Export
    Trf_cont_EVP_Exp1 = Trf_cont_EVP_Exp1 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
        2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Marchandises == "Conteneurs") & (
                df.Sens_ImpExpCab == "Export")].sum(axis=0)
    Trf_cont_EVP_Exp2 = Trf_cont_EVP_Exp2 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
        2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Nombre") & (
                    df.Marchandises == "Conteneurs") & (df.Sens_ImpExpCab == "Export")].sum(axis=0)
    # Export_vide
    Trf_cont_EVP_Exp_vid1 = Trf_cont_EVP_Exp_vid1 + ((df.apply(
        lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
            2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Marchandises == "Conteneurs") & (
                    df.Sens_ImpExpCab == "Export") & (df.Type_Conteneurs == "Vide")].sum(axis=0)
    Trf_cont_EVP_Exp_vid2 = Trf_cont_EVP_Exp_vid2 + ((df.apply(
        lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
            2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Nombre") & (
                    df.Marchandises == "Conteneurs") & (df.Sens_ImpExpCab == "Export") & (
                    df.Type_Conteneurs == "Vide")].sum(axis=0)
    # Cabotage
    Trf_cont_EVP_Cab1 = Trf_cont_EVP_Cab1 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
        2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Marchandises == "Conteneurs") & (
                    df.Sens_ImpExpCab == "Cabotage")].sum(axis=0)
    Trf_cont_EVP_Cab2 = Trf_cont_EVP_Cab2 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
        2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Nombre") & (
                df.Marchandises == "Conteneurs") & (df.Sens_ImpExpCab == "Cabotage")].sum(axis=0)

Tx_Cont_evp = (Trf_cont_EVP1 - Trf_cont_EVP2) / Trf_cont_EVP2
Tx_Cont_evp_Imp = (Trf_cont_EVP_Imp1 - Trf_cont_EVP_Imp2) / Trf_cont_EVP_Imp2
Tx_Cont_evp_Exp = (Trf_cont_EVP_Exp1 - Trf_cont_EVP_Exp2) / Trf_cont_EVP_Exp2
Tx_Cont_evp_Exp_vid = (Trf_cont_EVP_Exp_vid1 - Trf_cont_EVP_Exp_vid2) / Trf_cont_EVP_Exp_vid2
Tx_Cont_evp_Cab = (Trf_cont_EVP_Cab1 - Trf_cont_EVP_Cab2) / Trf_cont_EVP_Cab2

# st.write(Trf_cont_EVP1)
# st.write('{0:.1%}'.format(Tx_Cont_evp))
# st.write(Trf_cont_EVP_Imp1)
# st.write('{0:.1%}'.format(Tx_Cont_evp_Imp))
# st.write(Trf_cont_EVP_Exp1)
# st.write('{0:.1%}'.format(Tx_Cont_evp_Exp))
# st.write(Trf_cont_EVP_Exp_vid1)
# st.write('{0:.1%}'.format(Tx_Cont_evp_Exp_vid))
#
# st.write('{0:.1%}'.format(Trf_cont_EVP_Exp_vid1/Trf_cont_EVP_Exp1))

# st.write(Trf_cont_EVP_Cab1)
# st.write('{0:.1%}'.format(Tx_Cont_evp_Cab))

# '{:,}'.format(int(Trf_cont_EVP1)).replace(',',' ')
st.write("CONTENEURS")
if Trf_cont_EVP1 < 0:
    st.write('L’activité des conteneurs dans les ports relevant de l’ANP s’est chiffrée à ',
             '{:,}'.format(int(Trf_cont_EVP1)).replace(',', ' '), 'EVP à fin '
             , Mois2[Mois_Select - 1], Année_Select, ', soit une baisse de ', '{0:.1%}'.format(Tx_Cont_evp),
             'par rapport à la même période de l’année précédente.')
else:
    st.write('L’activité des conteneurs dans les ports relevant de l’ANP s’est chiffrée à ',
             '{:,}'.format(int(Trf_cont_EVP1)).replace(',', ' '), 'EVP à fin '
             , Mois2[Mois_Select - 1], Année_Select, ', soit une hausse de ', '{0:.1%}'.format(Tx_Cont_evp),
             'par rapport à la même période de l’année précédente.')

# présentation Trafic conteneurs
Trf_cont_Tn1 = 0
Trf_cont_Tn2 = 0
i = 0
for i in range(Mois_Select + 1):
    Trf_cont_Tn1 = Trf_cont_Tn1 + df['Valeurs'].loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Marchandises == "Conteneurs")].sum(
        axis=0)
    Trf_cont_Tn2 = Trf_cont_Tn2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (
                df.Marchandises == "Conteneurs")].sum(axis=0)

    Tx_Cont_tn = (Trf_cont_Tn1 - Trf_cont_Tn2) / Trf_cont_Tn2
# st.write(round(Trf_cont_Tn1/1000000,2))
# st.write('{0:.1%}'.format(Tx_Cont_tn))


if Tx_Cont_tn < 0:
    st.write('En tonnage, le trafic des conteneurs a marqué une baisse de ', '{0:.1%}'.format(Tx_Cont_tn),
             ', avec un volume de ', round(Trf_cont_Tn1 / 1000000, 1), 'millions de tonnes.')
else:
    st.write('En tonnage, le trafic des conteneurs a marqué une hausse de ', '{0:.1%}'.format(Tx_Cont_tn),
             ', avec un volume de ', round(Trf_cont_Tn1 / 1000000, 1), 'millions de tonnes.')
st.write('Par nature de flux, les évolutions enregistrées se présentent comme suit:')

if Tx_Cont_evp_Imp < 0:
    st.write('- Les importations ont atteint ', '{:,}'.format(int(Trf_cont_EVP_Imp1)).replace(',', ' '),
             'EVP, en baisse de ', '{0:.1%}'.format(Tx_Cont_evp_Imp),
             'par rapport à la même période de l’année précédente.')
else:
    st.write('- Les importations ont atteint ', '{:,}'.format(int(Trf_cont_EVP_Imp1)).replace(',', ' '),
             'EVP, en hausse de ', '{0:.1%}'.format(Tx_Cont_evp_Imp),
             'par rapport à la même période de l’année précédente.')

if Tx_Cont_evp_Exp < 0:
    st.write('- Les exportations ont enregistré une baisse de ', '{0:.1%}'.format(Tx_Cont_evp_Exp),
             'avec un volume de ', '{:,}'.format(int(Trf_cont_EVP_Exp1)).replace(',', ' '), 'EVP.')
else:
    st.write('- Les exportations ont enregistré une hausse de ', '{0:.1%}'.format(Tx_Cont_evp_Exp),
             'avec un volume de ', '{:,}'.format(int(Trf_cont_EVP_Exp1)).replace(',', ' '), 'EVP.')

st.write('Les conteneurs vides à l’export ont affiché un volume de ',
         '{:,}'.format(int(Trf_cont_EVP_Exp_vid1)).replace(',', ' '), '(', '{0:.1%}'.format(Tx_Cont_evp_Exp_vid),
         '), représentant ainsi ', '{0:.1%}'.format(Trf_cont_EVP_Exp_vid1 / Trf_cont_EVP_Exp1),
         'du trafic global des conteneurs à l’export.')

if Tx_Cont_evp_Cab < 0:
    st.write('- Le cabotage a connu une baisse de ', '{0:.1%}'.format(Tx_Cont_evp_Cab), 'avec un volume de ',
             '{:,}'.format(int(Trf_cont_EVP_Cab1)).replace(',', ' '), 'EVP.')
else:
    st.write('- Le cabotage a connu une hausse de ', '{0:.1%}'.format(Tx_Cont_evp_Cab), 'avec un volume de ',
             '{:,}'.format(int(Trf_cont_EVP_Cab1)).replace(',', ' '), 'EVP.')

# In[34]:


##st.write(df['Port'].loc[(df.Marchandises=="Conteneurs")&(df.Année.isin([(Année_Select -1),(Année_Select)]))].unique())


# In[35]:


# Représentation graphique courbe_ conteneurs
# df_Select=df.loc[df.Année.isin([(Année_Select -1),(Année_Select)])]
# Table_Select=(pd.pivot_table(df_Select ,index=['Port', 'Sous_Catégorie', 'Trafic',  'Marchandises', 'Marchandises_V0','Combined', 'Ind_PhosphatesDerives',  'Opérateur', 'Unité','Sens','Sens_ImpExp', 'Sens_ImpExpCab','Mois'], columns=['Année'],  values='Valeurs',aggfunc= np.sum,fill_value=0).reset_index())
# list_port_Cont1[Année_Select]=dict()
# list_port_Cont2[Année_Select-1]=dict()

list_port_Cont1 = dict()
list_port_Cont2 = dict()

i = 0
j = 0

for j in list(df['Port'].loc[
                  (df.Marchandises == "Conteneurs") & (df.Année.isin([(Année_Select - 1), (Année_Select)]))].unique()):
    list_port_Cont1[j] = 0
    list_port_Cont2[j] = 0
    port_cont1 = 0
    port_cont2 = 0
    for i in range(Mois_Select + 1):
        port_cont1 = port_cont1 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
            2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
            (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Port == j) & (
                        df.Marchandises == "Conteneurs")].sum(axis=0)
        port_cont2 = port_cont2 + ((df.apply(lambda x: 2 if x['Dimension_Conteneurs'] == '40 Pieds' else (
            2.25 if x['Dimension_Conteneurs'] == '45 Pieds' else 1), axis=1)) * df['Valeurs']).loc[
            (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Port == j) & (
                        df.Marchandises == "Conteneurs")].sum(axis=0)
        i = i + 1

    list_port_Cont1[j] = port_cont1
    list_port_Cont2[j] = port_cont2
##st.write(list_port_Cont1)
##st.write(list_port_Cont2)
##st.write(list(list_port_Cont2[key] for key in list_port_Cont1.keys()))


# In[36]:


x = np.arange(4)

width = 0.2

x1 = plt.bar(x - 0.1, list_port_Cont1.values(), width, color='green')
x2 = plt.bar(x + 0.1, (list(list_port_Cont2[key] for key in list_port_Cont1.keys())), width, color='orange')
plt.xticks(x, list_port_Cont1.keys())
# plt.xlabel("Teams")
# plt.ylabel("Scores")
# plt.legend(["Round 1", "Round 2", "Round 3"])
# plt.legend((Mois2[Mois_Select-1] ,(Année_Select-1)), (Mois2[Mois_Select-1] ,Année_Select))
# plt.legend((Année_Select-1), Année_Select)
plt.legend([x1, x2], [(Mois2[Mois_Select - 1], (Année_Select - 1)), (Mois2[Mois_Select - 1], Année_Select)])

# plt.title('........')

##plt.show()


# In[37]:


# st.write(dict(Port_Vol.remove(0)))
ContChart1 = dict()
ContChart1['Autres Ports'] = 0
for i, j in list_port_Cont1.items():
    if j in (dict(nlargest(2, list_port_Cont1.items(), itemgetter(1))).values()):
        ContChart1[i] = round(j / 1000, 1)
    elif j not in (dict(nlargest(2, list_port_Cont1.items(), itemgetter(1))).values()):
        ContChart1['Autres Ports'] = ContChart1['Autres Ports'] + (round(j / 1000, 1))
##st.write(ContChart1)
# st.write(dict(Port_Vol.remove(0)))
ContChart2 = dict()
ContChart2['Autres Ports'] = 0
for i, j in list_port_Cont2.items():
    if j in (dict(nlargest(2, list_port_Cont2.items(), itemgetter(1))).values()):
        ContChart2[i] = round(j / 1000, 1)
    elif j not in (dict(nlargest(2, list_port_Cont2.items(), itemgetter(1))).values()):
        ContChart2['Autres Ports'] = ContChart2['Autres Ports'] + (round(j / 1000, 1))
##st.write(ContChart2)
ContChart1_tri = dict(sorted(ContChart1.items(), key=lambda t: t[1], reverse=True))
##st.write(ContChart1_tri)
ContChart2_tri = dict(sorted(ContChart2.items(), key=lambda t: t[1], reverse=True))
##st.write(ContChart2_tri)

# Représentation graphique
x = np.arange(len(ContChart1_tri))

width = 0.2

fig, ax = plt.subplots(figsize=(7,3.5))
x1 = ax.bar(x - 0.1, (list(ContChart2_tri[key] for key in ContChart1_tri.keys())), width, color='green',
            label=(Année_Select - 1))
x2 = ax.bar(x + 0.1, ContChart1_tri.values(), width, color='orange', label=(Année_Select))
ax.set_xticks(x)
ax.set_xticklabels(ContChart1_tri.keys())
ax.legend()


def autolabel(bgraph):
    for bg in bgraph:
        height = bg.get_height()
        ax.text(bg.get_x() + bg.get_width() / 2., 1 * height, '%d' % int(height), ha='center',
                va='bottom')  # position du


# labelisation:
autolabel(x1)
autolabel(x2)
fig.tight_layout()
st.pyplot(fig)

# plt.xlabel("Teams")
# plt.ylabel("Scores")
# plt.legend(["Round 1", "Round 2", "Round 3"])
# plt.legend((Mois2[Mois_Select-1] ,(Année_Select-1)), (Mois2[Mois_Select-1] ,Année_Select))
# plt.legend((Année_Select-1), Année_Select)


# plt.legend([x1, x2], [(Mois2[Mois_Select-1] ,(Année_Select-1)), (Mois2[Mois_Select-1] ,Année_Select)])
# a=[(Mois2[Mois_Select-1] ,(Année_Select-1)), (Mois2[Mois_Select-1] ,Année_Select)]

# plt.bar_label(x1, padding=3)
# plt.bar_label(x2, padding=3)

# plt.title("Evolution du trafic des conteneurs (En EVP)")
# plt.ylim((0 if (min(y1)-15000)<5000 else 15000),(max(y1)+5000))
# plt.ylabel('En milliers de tonnes')

# plt.show()


# In[38]:


# Céréales
# for i in list(df['Port'].unique()):
#    st.write(i)


Trf_Cereal1 = 0
Trf_Cereal2 = 0

i = 0
for i in range(Mois_Select + 1):
    Trf_Cereal1 = Trf_Cereal1 + df['Valeurs'].loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Marchandises == "Céréales")].sum(
        axis=0)
    Trf_Cereal2 = Trf_Cereal2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (
                    df.Marchandises == "Céréales")].sum(axis=0)
    i = i + 1

Tx_Cereal = (Trf_Cereal1 - Trf_Cereal2) / Trf_Cereal2

# st.write(Trf_Cereal1)
# st.write(Tx_Cereal)
st.write("CEREALES")
if Tx_Cereal < 0:
    st.write('Les importations des céréales se sont chiffrées à', round(Trf_Cereal1 / 1000000, 1),
             'millions de tonnes, à fin', Mois2[Mois_Select - 1], Année_Select,
             ', marquant ainsi une baisse de', '{0:.1%}'.format(abs(Tx_Cereal)),
             'par rapport à la même période de l’année précédente.')
else:
    st.write('Les importations des céréales se sont chiffrées à', round(Trf_Cereal1 / 1000000, 1),
             'millions de tonnes, à fin', Mois2[Mois_Select - 1], Année_Select,
             ', marquant ainsi une hausse de', '{0:.1%}'.format(Tx_Cereal),
             'par rapport à la même période de l’année précédente.')

dic_port_Cereal1 = dict()
dic_port_Cereal2 = dict()

dic_port_Cereal_Tx = dict()
i = 0
j = 0

for j in list(df['Port'].loc[
                  (df.Marchandises == "Céréales") & (df.Année.isin([(Année_Select - 1), (Année_Select)]))].unique()):
    dic_port_Cereal1[j] = 0
    dic_port_Cereal2[j] = 0
    dic_port_Cereal_Tx[j] = 0
    port_Cereal1 = 0
    port_Cereal2 = 0
    for i in range(Mois_Select + 1):
        port_Cereal1 = port_Cereal1 + df['Valeurs'].loc[
            (df.Année == (Année_Select)) & (df.Mois == i) & (df.Port == j) & (df.Unité == "Tonne") & (
                        df.Marchandises == "Céréales")].sum(axis=0)
        port_Cereal2 = port_Cereal2 + df['Valeurs'].loc[
            (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Port == j) & (df.Unité == "Tonne") & (
                        df.Marchandises == "Céréales")].sum(axis=0)
        i = i + 1

    dic_port_Cereal1[j] = port_Cereal1
    dic_port_Cereal2[j] = port_Cereal2
    dic_port_Cereal_Tx[j] = (port_Cereal1 - port_Cereal2) / port_Cereal2

# st.write(dic_port_Cereal1)
# st.write(dic_port_Cereal2)
# st.write(dic_port_Cereal_Tx)
Port1_Cerl = dict(nlargest(1, dic_port_Cereal1.items(), itemgetter(1)))
dic_port_Cereal1_tri = dict(sorted(dic_port_Cereal1.items(), key=lambda t: t[1], reverse=True))
st.write('L’analyse de la répartition de ce trafic par port fait ressortir ce qui suit:')

if (dic_port_Cereal_Tx[[key for key in Port1_Cerl.keys()][0]]) < 0:
    st.write('- Une forte concentration de cette activité au port de', [key for key in Port1_Cerl.keys()][0],
             ', avec ', round([val for val in Port1_Cerl.values()][0] / 1000000, 1),
             'millions de tonnes, représentant environ,',
             '{0:.1%}'.format([val for val in Port1_Cerl.values()][0] / Trf_Cereal1),
             'du trafic global des céréales (soit une baisse de ',
             '{0:.1%}'.format(abs(dic_port_Cereal_Tx[[key for key in Port1_Cerl.keys()][0]])), ');')
else:
    st.write('- Une forte concentration de cette activité au port de', [key for key in Port1_Cerl.keys()][0],
             ', avec ', round([val for val in Port1_Cerl.values()][0] / 1000000, 1),
             'millions de tonnes, représentant environ,',
             '{0:.1%}'.format([val for val in Port1_Cerl.values()][0] / Trf_Cereal1),
             'du trafic global des céréales (soit une hausse de ',
             '{0:.1%}'.format(abs(dic_port_Cereal_Tx[[key for key in Port1_Cerl.keys()][0]])), ');')

st.write(
    '- Les importations en cette denrée dans les autres ports, ont enregistré des variations plus ou moins importantes, à savoir:',
    [key for key in dic_port_Cereal1_tri.keys()][1], '(',
    '{0:.1%}'.format(dic_port_Cereal_Tx[[key for key in dic_port_Cereal1_tri.keys()][1]]), '),',
    [key for key in dic_port_Cereal1_tri.keys()][2], '(',
    '{0:.1%}'.format(dic_port_Cereal_Tx[[key for key in dic_port_Cereal1_tri.keys()][2]]), '),',
    [key for key in dic_port_Cereal1_tri.keys()][3], '(',
    '{0:.1%}'.format(dic_port_Cereal_Tx[[key for key in dic_port_Cereal1_tri.keys()][3]]), '),',
    [key for key in dic_port_Cereal1_tri.keys()][4], '(',
    '{0:.1%}'.format(dic_port_Cereal_Tx[[key for key in dic_port_Cereal1_tri.keys()][4]]), ').')

# st.write([key for key in Port1_Cerl.keys()][0])
#    dict(nlargest(2, list_port_Cont1.items(), itemgetter(1))))
# for i in dic_port_Cereal_Tx.values(): st.write(i)


# In[39]:


# PieChart_tri=dict(sorted(PieChart.items(), key=lambda t: t[1], reverse=True))

explode = (0, 0.15, 0, 0)
#fig = plt.figure(1, figsize=(7, 7))
fig, ax = plt.subplots(figsize=(7,3.5))

# plt.pie(dict(Port_Vol).values(),labels=dict(Port_Vol).keys(), autopct='%1.1f%%', startangle=90, shadow=True)
a = ax.pie(dic_port_Cereal1_tri.values(), labels=dic_port_Cereal1_tri.keys(), autopct='%1.1f%%', startangle=90,
            shadow=True)
# ,explode=(0,0.1,0,0,0.1,0,0.2,0.6)
plt.axis('equal')
#st.pyplot(fig)

# In[40]:


# PieChart_tri=dict(sorted(PieChart.items(), key=lambda t: t[1], reverse=True))
dic_port_Cereal1_tri = dict(sorted(dic_port_Cereal1.items(), key=lambda t: t[1], reverse=True))
explode = (0, 0.15, 0, 0)
#fig = plt.figure(1, figsize=(7, 7))
fig, ax = plt.subplots(figsize=(7,3.5))

colors = ("orange", "green", "brown",
          "grey", "beige")
# plt.pie(dict(Port_Vol).values(),labels=dict(Port_Vol).keys(), autopct='%1.1f%%', startangle=90, shadow=True)
a = ax.pie(dic_port_Cereal1_tri.values(), labels=dic_port_Cereal1_tri.keys(), autopct='%1.1f%%', colors=colors,
            startangle=90, shadow=True, explode=(0, 0.1, 0, 0, 0.15))
# ,explode=(0,0.1,0,0,0.1,0,0.2,0.6)

# "beige"indigo,cyan
plt.axis('equal')
st.pyplot(fig)

# In[41]:


Trf_PhospDerv1 = 0
Trf_PhospDerv2 = 0

i = 0
for i in range(Mois_Select + 1):
    Trf_PhospDerv1 = Trf_PhospDerv1 + df['Valeurs'].loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Tonne") & (
            df.Ind_PhosphatesDerives.isin([1, 2, 3]))].sum(axis=0)
    Trf_PhospDerv2 = Trf_PhospDerv2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (
            df.Ind_PhosphatesDerives.isin([1, 2, 3]))].sum(axis=0)
    i = i + 1

Tx_PhospDerv = (Trf_PhospDerv1 - Trf_PhospDerv2) / Trf_PhospDerv2

# st.write(Trf_PhospDerv1)
# st.write(Tx_PhospDerv)
st.write("PHOSPHATES & DERIVES")

if Tx_PhospDerv < 0:
    st.write('Le trafic des phosphates et dérivées a atteint, à fin', Mois2[Mois_Select - 1], Année_Select,
             'un volume d’environ'
             , round(Trf_PhospDerv1 / 1000000, 1), 'millions de tonnes,',
             ' enregistrant une baisse de', '{0:.1%}'.format(abs(Tx_PhospDerv)),
             'par rapport à la même période de l’année précédente.')
else:
    st.write('Le trafic des phosphates et dérivées a atteint, à fin', Mois2[Mois_Select - 1], Année_Select,
             'un volume d’environ'
             , round(Trf_PhospDerv1 / 1000000, 1), 'millions de tonnes,',
             ' enregistrant une hausse de', '{0:.1%}'.format(Tx_PhospDerv),
             'par rapport à la même période de l’année précédente.')

##################

dic_PhospDerv_trfs1 = dict()
dic_PhospDerv_trfs2 = dict()

dic_PhospDerv_trfs_Tx = dict()
i = 0
j = 0

for j in list(df['Marchandises'].loc[(df.Unité == "Tonne") & (df.Ind_PhosphatesDerives.isin([1, 2, 3]))].unique()):

    #    dic_PhospDerv_trfs1[j]=0
    #   dic_PhospDerv_trfs2[j]=0
    dic_PhospDerv_trfs_Tx[j] = 0
    PhospDerv_trfs1 = 0
    PhospDerv_trfs2 = 0
    for i in range(Mois_Select + 1):
        PhospDerv_trfs1 = PhospDerv_trfs1 + df['Valeurs'].loc[
            (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Marchandises == j) & (
                df.Ind_PhosphatesDerives.isin([1, 2, 3]))].sum(axis=0)
        PhospDerv_trfs2 = PhospDerv_trfs2 + df['Valeurs'].loc[
            (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (df.Marchandises == j) & (
                df.Ind_PhosphatesDerives.isin([1, 2, 3]))].sum(axis=0)
        i = i + 1
    if PhospDerv_trfs1 == 0:
        continue
    dic_PhospDerv_trfs1[j] = int(round(PhospDerv_trfs1 / 1000, 1))
    dic_PhospDerv_trfs2[j] = int(round(PhospDerv_trfs2 / 1000, 1))
    dic_PhospDerv_trfs_Tx[j] = (PhospDerv_trfs1 - PhospDerv_trfs2) / PhospDerv_trfs2

st.write(dic_PhospDerv_trfs1)
# st.write(dic_PhospDerv_trfs2)
# st.write(dic_PhospDerv_trfs_Tx)
# Port1_Cerl=dict(nlargest(1, dic_port_Cereal1.items(), itemgetter(1)))
dic_PhospDerv_trfs1_tri = dict(sorted(dic_PhospDerv_trfs1.items(), key=lambda t: t[1], reverse=True))
dic_PhospDerv_trfs2_tri = dict(sorted(dic_PhospDerv_trfs2.items(), key=lambda t: t[1], reverse=True))

st.write('Cette évolution est due aux variations suivantes: '
         , [key for key in dic_PhospDerv_trfs1_tri.keys()][0], '(',
         '{0:.1%}'.format(dic_PhospDerv_trfs_Tx[[key for key in dic_PhospDerv_trfs1_tri.keys()][0]]), '),',
         [key for key in dic_PhospDerv_trfs1_tri.keys()][1], '(',
         '{0:.1%}'.format(dic_PhospDerv_trfs_Tx[[key for key in dic_PhospDerv_trfs1_tri.keys()][1]]), '),'
         , [key for key in dic_PhospDerv_trfs1_tri.keys()][2], '(',
         '{0:.1%}'.format(dic_PhospDerv_trfs_Tx[[key for key in dic_PhospDerv_trfs1_tri.keys()][2]]), '),',
         [key for key in dic_PhospDerv_trfs1_tri.keys()][3], '(',
         '{0:.1%}'.format(dic_PhospDerv_trfs_Tx[[key for key in dic_PhospDerv_trfs1_tri.keys()][3]]), '),'
         , [key for key in dic_PhospDerv_trfs1_tri.keys()][4], '(',
         '{0:.1%}'.format(dic_PhospDerv_trfs_Tx[[key for key in dic_PhospDerv_trfs1_tri.keys()][4]]), '),',
         [key for key in dic_PhospDerv_trfs1_tri.keys()][5], '(',
         '{0:.1%}'.format(dic_PhospDerv_trfs_Tx[[key for key in dic_PhospDerv_trfs1_tri.keys()][5]]), ').')

# st.write(dic_PhospDerv_trfs1_tri)
# st.write(dic_PhospDerv_trfs2_tri)

# Représentation graphique

# plt.figure(figsize=(15, 15))
x = np.arange(len(dic_PhospDerv_trfs1_tri))
# fig, ax = plt.subplots(figsize=(17, 9)) # cadre du graphique

width = 0.4
opacity = 0.6
fig, ax = plt.subplots(figsize=(7,3.5))
# x1=ax.bar(x-0.1, ContChart2_tri.values(), width,  color='green', label=(Année_Select-1))
# x2=ax.bar(x+0.1, ContChart1_tri.values(), width, color='orange', label=(Année_Select))
# fig = plt.figure(1, figsize=(10,7))
# figure(figsize=(30,25), dpi=120)

x1 = ax.bar(x - 0.2, (list(dic_PhospDerv_trfs2_tri[key] for key in dic_PhospDerv_trfs1_tri.keys())), width,
            align='center', alpha=opacity, color='green', label=(Année_Select - 1))
x2 = ax.bar(x + 0.2, dic_PhospDerv_trfs1_tri.values(), width, align='center', alpha=opacity, color='orange',
            label=(Année_Select))
# x3=ax.bar(x, dic_PhospDerv_trfs_Tx.values(), width, color='white')
plt.xticks(range(len(dic_PhospDerv_trfs2_tri)), dic_PhospDerv_trfs2_tri.keys(), rotation=30)
# ax.set_xticks(x)
# ax.set_xticklabels(dic_PhospDerv_trfs2_tri.keys(), rotation=30)

ax.legend()


def autolabel(bgraph):
    for bg in bgraph:
        height = bg.get_height()
        ax.text(bg.get_x() + bg.get_width() / 2., 1 * height, '%d' % int(height), ha='center',
                va='bottom')  # position du


# labelisation:
autolabel(x1)
autolabel(x2)

plt.ylabel('En milliers de tonnes')

plt.tight_layout()
plt.legend()
# plt.grid()

# plt.figure(figsize = (20,10))
st.pyplot(fig)

# st.write(df['Ind_PhosphatesDerives'].unique())
# st.write(df['Ind_PhosphatesDerives'].value_counts())
# st.write(df.Ind_PhosphatesDerives.unique())


# In[42]:


st.write("HYDROCARBURES")

Trf_Hydr1 = 0
Trf_Hydr2 = 0

i = 0
for i in range(Mois_Select + 1):
    Trf_Hydr1 = Trf_Hydr1 + df['Valeurs'].loc[(df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Tonne") & (
                df.Marchandises == "Hydrocarbures")].sum(axis=0)
    Trf_Hydr2 = Trf_Hydr2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Tonne") & (
                    df.Marchandises == "Hydrocarbures")].sum(axis=0)
    i = i + 1

Tx_Hydr = (Trf_Hydr1 - Trf_Hydr2) / Trf_Hydr2

# st.write(Trf_Hydr1)
# st.write(Tx_Hydr)


if Tx_Hydr < 0:
    st.write('Le trafic global des hydrocarbures s’est chiffré à', round(Trf_Hydr1 / 1000000, 1),
             'millions de tonnes, à fin', Mois2[Mois_Select - 1], Année_Select,
             ', soit une baisse de', '{0:.1%}'.format(abs(Tx_Hydr)),
             'par rapport à la même période de l’année précédente.')
else:
    st.write('Le trafic global des hydrocarbures s’est chiffré à', round(Trf_Hydr1 / 1000000, 1),
             'millions de tonnes, à fin', Mois2[Mois_Select - 1], Année_Select,
             ', soit une hausse de', '{0:.1%}'.format(Tx_Hydr), 'par rapport à la même période de l’année précédente.')

dic_port_Hydr1 = dict()
dic_port_Hydr2 = dict()

dic_port_Hydr_Tx = dict()
i = 0
j = 0

for j in list(df['Port'].loc[(df.Marchandises == "Hydrocarbures") & (
df.Année.isin([(Année_Select - 1), (Année_Select)]))].unique()):
    dic_port_Hydr1[j] = 0
    dic_port_Hydr2[j] = 0
    dic_port_Hydr_Tx[j] = 0
    port_Hydr1 = 0
    port_Hydr2 = 0
    for i in range(Mois_Select + 1):
        port_Hydr1 = port_Hydr1 + df['Valeurs'].loc[
            (df.Année == (Année_Select)) & (df.Mois == i) & (df.Port == j) & (df.Unité == "Tonne") & (
                        df.Marchandises == "Hydrocarbures")].sum(axis=0)
        port_Hydr2 = port_Hydr2 + df['Valeurs'].loc[
            (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Port == j) & (df.Unité == "Tonne") & (
                        df.Marchandises == "Hydrocarbures")].sum(axis=0)
        i = i + 1

    dic_port_Hydr1[j] = port_Hydr1
    dic_port_Hydr2[j] = port_Hydr2
    dic_port_Hydr_Tx[j] = (port_Hydr1 - port_Hydr2) / port_Hydr2

dic_port_Hydr1_tri = dict(sorted(dic_port_Hydr1.items(), key=lambda t: t[1], reverse=True))
# st.write(dic_port_Hydr1_tri)
# st.write(dic_port_Hydr_Tx)
st.write('L’analyse de ce trafic par port fait ressortir les principales variations suivantes: '
         , [key for key in dic_port_Hydr1_tri.keys()][0], '(',
         '{0:.1%}'.format(dic_port_Hydr_Tx[[key for key in dic_port_Hydr1_tri.keys()][0]]), '),',

         [key for key in dic_port_Hydr1_tri.keys()][1], '(',
         '{0:.1%}'.format(dic_port_Hydr_Tx[[key for key in dic_port_Hydr1_tri.keys()][1]]), '),'

         , [key for key in dic_port_Hydr1_tri.keys()][2], '(',
         '{0:.1%}'.format(dic_port_Hydr_Tx[[key for key in dic_port_Hydr1_tri.keys()][2]]), '),',

         [key for key in dic_port_Hydr1_tri.keys()][3], '(',
         '{0:.1%}'.format(dic_port_Hydr_Tx[[key for key in dic_port_Hydr1_tri.keys()][3]]), ').')

#     , [key for key in dic_port_Hydr1_tri.keys()][4],'(',
#    '{0:.1%}'.format(dic_port_Hydr_Tx[[key for key in dic_port_Hydr1_tri.keys()][4]]),'),',
#   [key for key in dic_port_Hydr1_tri.keys()][5],'(',
#  '{0:.1%}'.format(dic_port_Hydr_Tx[[key for key in dic_port_Hydr1_tri.keys()][5]]),').')
#
#
########### Analyse du cabotage

dic_port_Hydr_Cab1 = dict()
dic_port_Hydr_Cab2 = dict()

dic_port_Hydr_Cab_Tx = dict()
i = 0
j = 0

for j in list(df['Port'].loc[(df.Marchandises == "Hydrocarbures") & (df.Sens_ImpExpCab == "Cabotage") & (
df.Année.isin([(Année_Select - 1), (Année_Select)]))].unique()):
    dic_port_Hydr_Cab1[j] = 0
    dic_port_Hydr_Cab2[j] = 0
    dic_port_Hydr_Cab_Tx[j] = 0
    port_Hydr_Cab1 = 0
    port_Hydr_Cab2 = 0
    for i in range(Mois_Select + 1):
        port_Hydr_Cab1 = port_Hydr_Cab1 + df['Valeurs'].loc[
            (df.Année == (Année_Select)) & (df.Sens_ImpExpCab == "Cabotage") & (df.Mois == i) & (df.Port == j) & (
                        df.Unité == "Tonne") & (df.Marchandises == "Hydrocarbures")].sum(axis=0)
        port_Hydr_Cab2 = port_Hydr_Cab2 + df['Valeurs'].loc[
            (df.Année == (Année_Select - 1)) & (df.Sens_ImpExpCab == "Cabotage") & (df.Mois == i) & (df.Port == j) & (
                        df.Unité == "Tonne") & (df.Marchandises == "Hydrocarbures")].sum(axis=0)
        i = i + 1

    dic_port_Hydr_Cab1[j] = port_Hydr_Cab1
    dic_port_Hydr_Cab2[j] = port_Hydr_Cab2
    dic_port_Hydr_Cab_Tx[j] = (port_Hydr_Cab1 - port_Hydr_Cab2) / port_Hydr_Cab2
st.write(dic_port_Hydr_Cab1)

# In[43]:


##st.write(dict(nlargest(4, dic_port_Hydr1.items(), itemgetter(1)))['Mohammedia'])


# In[59]:


# Représentation graphique : Hydrocarbures
# 1. Regrouper d'abord les petits volumes
##st.write("Répartition du trafic global des hydrocarbures par port")
Dict_hydr_chart1 = dict()
Dict_hydr_chart1['Autres Ports'] = 0
for i, j in dic_port_Hydr1.items():
    if j in (dict(nlargest(4, dic_port_Hydr1.items(), itemgetter(1))).values()):
        Dict_hydr_chart1[i] = round(j / 1000, 1)
    elif j not in (dict(nlargest(4, dic_port_Hydr1.items(), itemgetter(1))).values()):
        Dict_hydr_chart1['Autres Ports'] = Dict_hydr_chart1['Autres Ports'] + (round(j / 1000, 1))
##st.write(Dict_hydr_chart1)
# st.write(dict(Port_Vol.remove(0)))
Dict_hydr_chart2 = dict()
Dict_hydr_chart2['Autres Ports'] = 0
for i, j in dic_port_Hydr2.items():
    if j in (dict(nlargest(4, dic_port_Hydr2.items(), itemgetter(1))).values()):
        Dict_hydr_chart2[i] = round(j / 1000, 1)
    elif j not in (dict(nlargest(4, dic_port_Hydr2.items(), itemgetter(1))).values()):
        Dict_hydr_chart2['Autres Ports'] = Dict_hydr_chart2['Autres Ports'] + (round(j / 1000, 1))
##st.write(Dict_hydr_chart2)

Dict_hydr_chart1_tri = dict(sorted(Dict_hydr_chart1.items(), key=lambda t: t[1], reverse=True))
##st.write(Dict_hydr_chart1_tri)
Dict_hydr_chart2_tri = dict(sorted(Dict_hydr_chart2.items(), key=lambda t: t[1], reverse=True))
##st.write(Dict_hydr_chart2_tri)

##st.write("Répartition du trafic cabotage des hydrocarbures par port")

Dict_hydr_Cab_chart1 = dict()
Dict_hydr_Cab_chart1['Autres Ports'] = 0
for i, j in dic_port_Hydr_Cab1.items():
    if j in {dic_port_Hydr_Cab1['Mohammedia'], dic_port_Hydr_Cab1['Jorf Lasfar'], dic_port_Hydr_Cab1['Agadir']}:
        Dict_hydr_Cab_chart1[i] = round(j / 1000, 1)
    elif j not in {dic_port_Hydr_Cab1['Mohammedia'], dic_port_Hydr_Cab1['Jorf Lasfar'], dic_port_Hydr_Cab1['Agadir']}:
        Dict_hydr_Cab_chart1['Autres Ports'] = Dict_hydr_Cab_chart1['Autres Ports'] + (round(j / 1000, 1))
##st.write(Dict_hydr_Cab_chart1)
# st.write(dict(Port_Vol.remove(0)))
Dict_hydr_Cab_chart2 = dict()
Dict_hydr_Cab_chart2['Autres Ports'] = 0
for i, j in dic_port_Hydr_Cab2.items():
    if j in {dic_port_Hydr_Cab2['Mohammedia'], dic_port_Hydr_Cab2['Jorf Lasfar'], dic_port_Hydr_Cab2['Agadir']}:
        Dict_hydr_Cab_chart2[i] = round(j / 1000, 1)
    elif j not in {dic_port_Hydr_Cab2['Mohammedia'], dic_port_Hydr_Cab2['Jorf Lasfar'], dic_port_Hydr_Cab2['Agadir']}:
        Dict_hydr_Cab_chart2['Autres Ports'] = Dict_hydr_Cab_chart2['Autres Ports'] + (round(j / 1000, 1))
##st.write(Dict_hydr_Cab_chart2)

Dict_hydr_Cab_chart1_tri = dict(sorted(Dict_hydr_Cab_chart1.items(), key=lambda t: t[1], reverse=True))
##st.write(Dict_hydr_Cab_chart1_tri)
Dict_hydr_Cab_chart2_tri = dict(sorted(Dict_hydr_Cab_chart2.items(), key=lambda t: t[1], reverse=True))
##st.write(Dict_hydr_Cab_chart2_tri)


# In[61]:


# Représentation graphique, plt.subplot(1,2,1)
st.write("Répartition du trafic global des hydrocarbures par port")
x = np.arange(len(Dict_hydr_chart1_tri))

width = 0.4
opacity = 0.6
fig, ax = plt.subplots(figsize=(7,3.5))
# fig,ax=plt.subplots(figsize=(10, 5)),plt.subplot(1,2,1)
# plt.subplot(1,2,1)
# (list(Dict_hydr_chart2_tri[key] for key in Dict_hydr_chart1_tri.keys()))
x1 = ax.bar(x - 0.2, (list(Dict_hydr_chart2_tri[key] for key in Dict_hydr_chart1_tri.keys())), width, align='center',
            alpha=opacity, color='green', label=(Année_Select - 1))
x2 = ax.bar(x + 0.2, Dict_hydr_chart1_tri.values(), width, align='center', alpha=opacity, color='orange',
            label=(Année_Select))
ax.set_xticks(x)
ax.set_xticklabels(Dict_hydr_chart1_tri.keys())
ax.legend()


def autolabel(bgraph):
    for bg in bgraph:
        height = bg.get_height()
        ax.text(bg.get_x() + bg.get_width() / 2., 1 * height, '%d' % int(height), ha='center',
                va='bottom')  # position du


# labelisation:
autolabel(x1)
autolabel(x2)
fig.tight_layout()
st.pyplot(fig)

# Graphique Cabotage, plt.subplot(1,2,2)
######
st.write("Répartition du trafic cabotage des hydrocarbures par port")

x02 = np.arange(len(Dict_hydr_Cab_chart1_tri))

width = 0.4
opacity = 0.6
fig02, ax02 = plt.subplots(figsize=(7,3.5))
# fig02,ax02=plt.subplots(figsize=(10, 5)),subplot(1,2,2)
# plt.subplot(1,2,2)
x3 = ax02.bar(x02 - 0.2, (list(Dict_hydr_Cab_chart2_tri[key] for key in Dict_hydr_Cab_chart1_tri.keys())), width,
              align='center', alpha=opacity, color='green', label=(Année_Select - 1))
x4 = ax02.bar(x02 + 0.2, Dict_hydr_Cab_chart1_tri.values(), width, align='center', alpha=opacity, color='orange',
              label=(Année_Select))
ax02.set_xticks(x02)
ax02.set_xticklabels(Dict_hydr_Cab_chart1_tri.keys())
ax02.legend()


def autolabel1(bgraph):
    for bg in bgraph:
        height = bg.get_height()
        ax02.text(bg.get_x() + bg.get_width() / 2., 1 * height, '%d' % int(height), ha='center',
                  va='bottom')  # position du


# labelisation:
autolabel1(x3)
autolabel1(x4)
fig02.tight_layout()
st.pyplot(fig)

# In[46]:


##st.write(list(Dict_hydr_Cab_chart2_tri[key] for key in Dict_hydr_Cab_chart1_tri.keys()))
##st.write(list(key for key in Dict_hydr_Cab_chart1_tri.keys()))

##st.write(Dict_hydr_Cab_chart1_tri.keys())
##st.write(Dict_hydr_Cab_chart1_tri.values())
# st.write(list(value for value in Dict_hydr_Cab_chart2_tri[Dict_hydr_Cab_chart1_tri.keys()]))
# st.write(list(value for value in Dict_hydr_Cab_chart2_tri[list(key for key in Dict_hydr_Cab_chart1_tri.keys())]))
# st.write(sorted(Dict_hydr_Cab_chart2_tri.items(), key=lambda t: t[0][(key for key in Dict_hydr_Cab_chart1_tri.keys())]))


# In[47]:


st.write("TIR")

Trf_TIR1 = 0
Trf_TIR2 = 0
Trf_TIR_Nador1 = 0

i = 0
for i in range(Mois_Select + 1):
    Trf_TIR1 = Trf_TIR1 + df['Valeurs'].loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Marchandises == "TIR")].sum(axis=0)
    Trf_TIR2 = Trf_TIR2 + df['Valeurs'].loc[
        (df.Année == (Année_Select - 1)) & (df.Mois == i) & (df.Unité == "Nombre") & (df.Marchandises == "TIR")].sum(
        axis=0)
    Trf_TIR_Nador1 = Trf_TIR_Nador1 + df['Valeurs'].loc[
        (df.Année == (Année_Select)) & (df.Mois == i) & (df.Port == "Nador") & (df.Unité == "Nombre") & (
                    df.Marchandises == "TIR")].sum(axis=0)

    i = i + 1

Tx_TIR = (Trf_TIR1 - Trf_TIR2) / Trf_TIR2

# st.write(Trf_TIR1)
# st.write(Tx_TIR)
'{:,}'.format(int(Trf_cont_EVP_Imp1)).replace(',', ' '),

if Tx_TIR < 0:
    st.write('Le trafic TIR a atteint, à fin', Mois2[Mois_Select - 1], Année_Select, 'un volume de ',
             '{:,}'.format(int(Trf_TIR1)).replace(',', ' '), 'unités, '
                                                             ', soit une baisse de', '{0:.1%}'.format(abs(Tx_TIR)),
             'par rapport à la même période de l’année précédente.')
else:
    st.write('Le trafic TIR a atteint, à fin', Mois2[Mois_Select - 1], Année_Select, 'un volume de ',
             '{:,}'.format(int(Trf_TIR1)).replace(',', ' '), 'unités, '
                                                             ', soit une hausse de', '{0:.1%}'.format(Tx_TIR),
             'par rapport à la même période de l’année précédente.')
st.write('Le port de Nador a assuré, à lui seul,', '{:,}'.format(int(Trf_TIR_Nador1)).replace(',', ' '), 'unités.')
