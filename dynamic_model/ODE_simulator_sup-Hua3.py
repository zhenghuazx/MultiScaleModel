import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import minimize, Parameters, Parameter, report_fit,  fit_report
from scipy.integrate import odeint
from scipy import interpolate
import scipy
import time
import pickle
###Fixed Parameter with value
# vmaxPFKALD = 2.40E-03
# vmaxPGK = 5.00E-04
# vmaxTKTA = 2.00E-05
# vmaxGluT = 1.28E-06
# vmaxleak = 8.70E-05
# vmaxfCK = 2.30E-04
# vmaxrCK = 9.00E-05
# vmaxfAK = 1.00E-04
# vmaxrAK = 3.00E-05
# vmaxPPRibP = 6.00E-10
# vmaxNADPHox = 4.00E-03
# vmaxrASTA = 3.25E-06


# K_mGLC = 4.19E+00
# K_mGLN = 1.58E+00
# K_mLAC = 5.00E+00
# K_mNH4 = 5.00E-01
# K_mALA = 1.00E+00
# K_mARG = 5.00E-02
# K_mASP = 1.50E-02
# K_mASX = 1.50E-01
K_mGLY = 5.00E-02
# K_mHIS = 5.00E-01
# K_mILE = 2.50E-02
# K_mLUE = 2.50E-02
# K_mLYS = 5.00E-02
# K_mSER = 1.50E-02
# K_mTYR = 5.00E-02
# K_mVAL = 5.00E-02
# K_aGLN = 1.00E+01
# K_mGLNmAb = 1.00E-02
# K_mG6P = 8.20E-08
# K_mF6P = 4.00E-07
# K_mGAP = 8.00E-07
# K_mPEP = 1.70E-07
# K_mRu5P = 1.30E-08
# K_mX5P = 3.40E-08
# K_mPYR = 9.80E-07
# K_mAcCoA = 9.60E-07
# K_mOAA = 2.40E-07
# K_mCIT = 1.00E-07
# K_mAKG = 8.60E-07
# K_mSUC = 2.80E-08
# K_mMAL = 8.50E-07
# K_mGLU = 1.28E-04
# K_mATP = 4.20E-06
# K_mADP = 3.65E-07
# K_mAMP = 2.52E-08
# K_mPCr = 7.70E-06
# K_mCr = 6.00E-07
# K_mO2 = 4.00E-06
# K_mPi = 1.00E-06
# K_mNADH = 1.00E-07
# K_mNADPH = 1.00E-09
# K_mATPtoADP = 1.00E+00
# K_mADPtoATP = 9.00E-06
# K_mNADHtoNAD = 3.50E-03
# K_mNADtoNADH = 5.00E-01
# K_mNADPtoNADPH = 1.00E-03
# K_iG6P = 5.12E-08
# K_iPEP = 2.00E-07
# K_iPYR = 3.00E-08
# K_aF6P = 1.00E-06
# vmaxresp=7.00E-03*1E3
# vmaxATPase=3.60E-03*1E3
# vmaxleak=8.70E-04*1E3
# vmaxfAK=2.30E-03*1E3
# vmaxrAK=9.00E-04*1E3
# vmaxfCK=1.00E-03*1E3
# vmaxrCK=3.00E-04*1E3
# vmaxPPRibP=6.00E-09*1E3
# vmaxNADPHox=4.00E-02*1E3
# vmaxG6PDHPGLcDH=7.88E-05*1E3
# vmaxEP=7.88E-05*1E3
# vmaxTKTA=7.88E-05*1E3

S1_0 = np.array([1.40E-07, 4.00E-07, 2.00E-07, 8.00E-06, 3.00E-07,
5.00E-08, 4.00E-07, 7.00E-04, 8.00E-06, 1E-6, 5.50E-03, 
4.00E-08, 5.00E-05, 1.50E-07, 5.30E-07, 5.30E-07, 5.30E-07, 5.30E-07,
5.30E-07, 5.30E-07])

###Stoichiometry Matrix
N = pd.read_excel('F:\document\PHD\iPSC\Simulator_v3.xlsx', sheet_name="N")
# Meta_list = N.columns[1:]
# .tolist()
# Meta_list = Meta_list.append()
# flux_list = ['HK', 'PGI', 'PEKALD', 'PGK', 'PK', 'LDH', 'G6PDHPGLcDH', 'EP',
#        'TKTA', 'PDH', 'CS', 'CITSISOD', 'AKGDH', 'SDHFUM', 'MLD', 'ME', 'PC',
#        'GLNS', 'GLDH', 'AlaTA', 'GluT', 'resp', 'leak', 'ATPase', 'AK', 'CK',
#        'PPRibP', 'NADPHox', 'SAL', 'ASX', 'ASTA', 'AA1', 'AA2', 'Bio', 'Cell']

flux_list = ['HK', 'PGI', 'PEKALD', 'PGK', 'PK', 'LDHf', 'LDHr', 'PyrT',
       'LacTf', 'LacTr', 'OP', 'NOP', 'PDH', 'CS', 'CITSISODf', 'CITSISODr', 'AKGDH',
       'SDH', 'FUMf', 'FUMr', 'MLDf', 'MLDr', 'ME', 'PC', 'GLNSf', 'GLNSr',
       'GLDHf', 'GLDHr', 'AlaTAf', 'AlaTAr', 'AlaT', 'GluT', 'GlnT', 
       'SAL', 'ASTAf', 'ASTAr', 'ASPT', 'ACL', 'Bio', 'Cell']

																																						

# meta_list = N.iloc[:, 0].tolist()


# meta_list = ['AcCoA', 'AKG', 'ADP', 'AMP', 'ATP', 'CIT', 'CoA', 'Cr', 'F6P', 'G6P', 'GAP',
#  'GLU', 'GLY', 'MAL', 'NAD', 'NADH', 'NADP', 'NADPH', 'OXA', 'O2', 'PEP', 'PCr',
#  'Pi', 'ARG', 'R5P', 'SUC', 'X5P', 'PYR', 'ALA', 'ASP', 'ASX', 'GLY', 'HIS',
#  'ILE', 'LEU', 'LYS', 'SER', 'TYR', 'VAL', 'GLC', 'GLN', 'EGLU', 'LAC', 'NH4',
#  'BIO', 'Cell']

meta_list = ['AcCoA', 'AKG', 'CIT', 'CO2', 'F6P', 'G6P', 'GAP', 'GLU', 'GLY', 'MAL', 'OAA',
 'PEP', 'FUM', 'Ru5P', 'SUC', 'PYR', 'ALA', 'ASP', 'LAC', 'GLN', 'GLY', 'SER',
 'GLC', 'EGLN', 'EGLU', 'EPYR', 'EASP', 'EALA', 'ELAC', 'NH4', 'LIPID', 'BIO', 'Cell']




N = N.fillna(0).values
N = N[:,1:]
N = N.astype(float).reshape(32,39)
# N = np.delete(N, (2,3,4), 0)
# N = np.delete(N, (23,24,25), 1)

data_extra_raw = pd.read_excel('F:\document\PHD\iPSC\iPSC_data.xlsx', sheet_name="Extra")
data_extra = data_extra_raw.values

#####19 observable state: EPYR, EALA, EASP, EASX, GLY, HIS, ILE, LUE, LYS, SER, TYR, VAL, GLC, EGLN, EGLU, ELAC, NH4, BIO, X
# data_extra = np.vstack((data_extra[:,6],data_extra[:,10],data_extra[:,12], data_extra[:,11]+data_extra[:,12], data_extra[:,13:18].T, data_extra[:,21], data_extra[:,24:26].T, data_extra[:,4], data_extra[:,7:9].T, data_extra[:,5], data_extra[:,9], data_extra[:,26], data_extra[:,3]))

#13 observable state: GLY, SER, GLC, EGLN, EGLU, EPYR, EASP, EALA, ELAC, NH4, LIPID, Bio, X

data_extra = np.vstack((data_extra[:,13], data_extra[:,21], data_extra[:,4], data_extra[:,7], data_extra[:,8], data_extra[:,6], 
                        data_extra[:,12], data_extra[:,10], data_extra[:,5], data_extra[:,9], data_extra[:,27], data_extra[:,26], data_extra[:,3] ))
data_extra = data_extra.T
data_extra = np.array(data_extra, dtype = float)

# Measurements under different DoE
data_extra_HGHL = data_extra[:30,:]
data_extra_HGLL = data_extra[30:60,:]
data_extra_LGLL = data_extra[60:90,:]
data_extra_LGHL = data_extra[90:,:]
data_whole = [data_extra_HGHL, data_extra_HGLL, data_extra_LGLL, data_extra_LGHL]

# Measurement Time
t = [0,12,24,36,48]
size = len(t)

# mean and measurement std for each metabolite at each time point 
mean_HGHL = np.zeros([5, 13])
mean_HGLL = np.zeros([5, 13])
mean_LGLL = np.zeros([5, 13])
mean_LGHL = np.zeros([5, 13])

std_HGHL = np.zeros([5, 13])
std_HGLL = np.zeros([5, 13])
std_LGLL = np.zeros([5, 13])
std_LGHL = np.zeros([5, 13])

for i in range(0, len(t)):
    index = np.arange(i,data_extra_HGHL[:,1].size,5)   
    mean_HGHL[i] = np.mean(data_extra_HGHL[index,],axis = 0)
    mean_HGLL[i] = np.mean(data_extra_HGLL[index,],axis = 0)
    mean_LGLL[i] = np.mean(data_extra_LGLL[index,],axis = 0)
    mean_LGHL[i] = np.mean(data_extra_LGHL[index,],axis = 0)
    
    std_HGHL[i] = np.std(data_extra_HGHL[index,],axis = 0)
    std_HGLL[i] = np.std(data_extra_HGLL[index,],axis = 0)
    std_LGLL[i] = np.std(data_extra_LGLL[index,],axis = 0)
    std_LGHL[i] = np.std(data_extra_LGHL[index,],axis = 0)

mean_whole = [mean_HGHL, mean_HGLL, mean_LGLL, mean_LGHL]
std_whole = [std_HGHL, std_HGLL, std_LGLL, std_LGHL]

# data_48_raw = pd.read_excel('F:\document\PHD\iPSC\iPSC_data.xlsx', sheet_name="Flux48")
# data_48 = data_48_raw.values
# data_48 = data_48[0:17,0:9]
# mea_flux = np.array(data_48[:,0],dtype=int) -1
# flux_48 = np.array(data_48[0:17,1:5], dtype = float)
# flux_std_48 = np.array(data_48[0:17,5:9], dtype = float)


#################################################################################################################################
#############################################Cell Growth Rate Estimation###########################################
#################################################################################################################################
def f_x(t, xs, ps):
    mu = ps['mu'].value
    X = xs
    
    dx = mu*X
    return dx


def g_x(t, x0, ps, t_eval=None):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    # x = odeint(f, x0, t, args=(ps,))
    start = time.time()
    x = scipy.integrate.solve_ivp(f_x, t,  x0, args=(ps,), t_eval=t_eval, method ='RK45' )
    end = time.time()
    print('Solving took: ' + str((end-start)) + ' sec')
    return x

    

def residual_x(ps, ts, data):
    sum_ls = []
    for i in range(0, len(data)):
        for r in range(0, 6):
            batch_data = data[i][(size*r):(size*r+5)]
            # print(batch_data)
            # batch_data = np.vstack((batch_data[:,6],batch_data[:,10],batch_data[:,12], batch_data[:,11]+batch_data[:,12], batch_data[:,13:18].T, batch_data[:,21], batch_data[:,24:26].T, batch_data[:,4], batch_data[:,7:9].T, batch_data[:,5], batch_data[:,9], batch_data[:,26], batch_data[:,3]))
            # batch_data = batch_data.astype(float)
            # x0 = np.hstack([S1_0*1000*batch_data[-1,0], batch_data[:,0]])
            x0 = np.array([batch_data[0]])
            model = g_x([ts[0], ts[-1]], x0, ps, t_eval=ts)

            sum_ls.append((model.y - batch_data).ravel())

    residuals = np.concatenate(sum_ls)
    # remove NaN
    residuals = residuals[~np.isnan(residuals)]
    return residuals

    # # residuals = np.concatenate(sum_ls)
    # # remove NaN
    # residuals = (model.y.T[:,24:] - batch_data.T).flatten()
    # residuals = residuals[~np.isnan(residuals)]
    # # print(residuals)
    # return residuals

params = Parameters()
params.add('mu', value=0.044, min=0, max=0.1)

#############################
# Train 0:HGHL, 1:HGLL, 2:LGLL, 
# Test 3:LGHL

indices = [0,1,2]
data_train = [data_whole[index][:,-1] for index in indices]
data_test = data_whole[3][:,-1]

result = minimize(residual_x, params, args=(t, data_train), method='leastsq')
report_fit(result)

parameters = result.params
#LGHL
x0_LGHL = np.array([np.mean(data_test[0])])
pred_LGHL = g_x([t[0], t[-1]], x0_LGHL, parameters, t).y  # g(t, x0_LGHL, parameters)

fig, ax = plt.subplots()

ax.plot([0,12,24,36,48],mean_whole[3][:,-1], 'o', color='blue', label='Measured')
ax.plot([0,12,24,36,48],pred_LGHL[0], color='black', label='Predicted')
ax.set_title('Cell Density (LGHL)')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Cell Density (10^6 cells/mL)')
ax.set_ylim(0,0.35)
plt.xticks([0,12,24,36,48])
leg = ax.legend()

plt.show()



#############################
# Train 1:HGLL, 2:LGLL, 3:LGHL
# Test 0:HGHL,

indices = [1,2,3]
data_train = [data_whole[index][:,-1] for index in indices]
data_test = data_whole[0][:,-1]

result = minimize(residual_x, params, args=(t, data_train), method='leastsq')
report_fit(result)

parameters = result.params
#HGHL
x0_HGHL = np.array([np.mean(data_test[0])])
pred_HGHL = g_x([t[0], t[-1]], x0_HGHL, parameters, t).y #  g(t, x0_HGHL, parameters)

fig, ax = plt.subplots()

ax.plot([0,12,24,36,48],mean_whole[0][:,-1], 'o', color='blue', label='Measured')
ax.plot([0,12,24,36,48],pred_HGHL[0], color='black', label='Predicted')
ax.set_title('Cell Density (HGHL)')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Cell Density (10^6 cells/mL)')
ax.set_ylim(0,0.35)
plt.xticks([0,12,24,36,48])
leg = ax.legend()

plt.show()

#############################
# Train 0:HGHL, 2:LGLL, 3:LGHL
# Test 1:HGLL, 

indices = [0,2,3]
data_train = [data_whole[index][:,-1] for index in indices]
data_test = data_whole[1][:,-1]

result = minimize(residual_x, params, args=(t, data_train), method='leastsq')
report_fit(result)

parameters = result.params
#HGLL
x0_HGLL = np.array([np.mean(data_test[0])])
pred_HGLL = g_x([t[0], t[-1]], x0_HGLL, parameters, t).y # g(t, x0_HGLL, parameters)

fig, ax = plt.subplots()

ax.plot([0,12,24,36,48],mean_whole[1][:,-1], 'o', color='blue', label='Measured')
ax.plot([0,12,24,36,48],pred_HGLL[0], color='black', label='Predicted')
ax.set_title('Cell Density (HGLL)')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Cell Density (10^6 cells/mL)')
ax.set_ylim(0,0.35)
plt.xticks([0,12,24,36,48])
leg = ax.legend()

plt.show()


#############################
# Train 0:HGHL, 1:HGLL, 3:LGHL
# Test 2:LGLL,

indices = [0,1,3]
data_train = [data_whole[index][:,-1] for index in indices]
data_test = data_whole[2][:,-1]

result = minimize(residual_x, params, args=(t, data_train), method='leastsq')
report_fit(result)

parameters = result.params
#LGLL
x0_LGLL = np.array([np.mean(data_test[0])])
pred_LGLL = g_x([t[0], t[-1]], x0_LGLL, parameters, t).y  # g(t, x0_LGLL, parameters)

fig, ax = plt.subplots()

ax.plot([0,12,24,36,48],mean_whole[2][:,-1], 'o', color='blue', label='Measured')
ax.plot([0,12,24,36,48],pred_LGLL[0], color='black', label='Predicted')
ax.set_title('Cell Density (LGLL)')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Cell Density (10^6 cells/mL)')
ax.set_ylim(0,0.35)
plt.xticks([0,12,24,36,48])
leg = ax.legend()

plt.show()


#############################
# Train 0:HGHL, 1:HGLL, 3:LGHL, 2:LGLL,

indices = [0,1,2,3]
data_train = [data_whole[index][:,-1] for index in indices]
# data_test = data_whole[2][:,-1]

result = minimize(residual_x, params, args=(t, data_train), method='leastsq')
report_fit(result)

###Estimated growth rate
parameters = result.params
mu_fit = parameters['mu'].value

##############################################################################################################
###############################################Flux Rate Estimation###########################################
#############################################################################################################
def f_m(t, xs, ps, mode = 1):
    """
    Bio_Kinetic Model.
    mode: 1: Return du of each metabolite
          0: Return the flux rate of each reaction
    """ 
    # 
    vmaxHK = ps['vmaxHK'].value
    vmaxPGI = ps['vmaxPGI'].value
    vmaxPFKALD = ps['vmaxPFKALD'].value
    vmaxPGK = ps['vmaxPGK'].value
    vmaxPK = ps['vmaxPK'].value
    vmaxfLDH = ps['vmaxfLDH'].value
    vmaxrLDH = ps['vmaxrLDH'].value
    vmaxPYRT = ps['vmaxPYRT'].value
    # vmaxG6PDHPGLcDH = ps['vmaxG6PDHPGLcDH'].value
    # vmaxEP = ps['vmaxEP'].value
    # vmaxTKTA = ps['vmaxTKTA'].value
    vmaxfLACT = ps['vmaxfLACT'].value
    vmaxrLACT = ps['vmaxrLACT'].value
    
    
    vmaxOP = ps['vmaxOP'].value
    vmaxNOP = ps['vmaxNOP'].value
    
    vmaxPDH = ps['vmaxPDH'].value
    vmaxCS = ps['vmaxCS'].value
    vmaxfCITSISOD = ps['vmaxfCITSISOD'].value
    vmaxrCITSISOD = ps['vmaxrCITSISOD'].value
    vmaxAKGDH = ps['vmaxAKGDH'].value
    vmaxSDH = ps['vmaxSDH'].value
    vmaxfFUM = ps['vmaxfFUM'].value
    vmaxrFUM = ps['vmaxrFUM'].value
    vmaxfMDH = ps['vmaxfMDH'].value
    vmaxrMDH = ps['vmaxrMDH'].value
    
    vmaxME = ps['vmaxME'].value
    vmaxPC = ps['vmaxPC'].value
    vmaxfGLNS = ps['vmaxfGLNS'].value
    vmaxrGLNS = ps['vmaxrGLNS'].value
    vmaxfGLDH = ps['vmaxfGLDH'].value
    vmaxrGLDH = ps['vmaxrGLDH'].value
    vmaxfAlaTA = ps['vmaxfAlaTA'].value
    vmaxrAlaTA = ps['vmaxrAlaTA'].value
    vmaxAlaT = ps['vmaxAlaT'].value
    vmaxGluT = ps['vmaxGluT'].value
    vmaxGlnT = ps['vmaxGlnT'].value
    # vmaxresp = ps['vmaxresp'].value
    # vmaxATPase = ps['vmaxATPase'].value
    # vmaxleak = ps['vmaxleak'].value
    # vmaxfAK = ps['vmaxfAK'].value
    # vmaxrAK = ps['vmaxrAK'].value
    # vmaxfCK = ps['vmaxfCK'].value
    # vmaxrCK = ps['vmaxrCK'].value
    # vmaxPPRibP = ps['vmaxPPRibP'].value
    # vmaxNADPHox = ps['vmaxNADPHox'].value
    vmaxSAL = ps['vmaxSAL'].value
    # vmaxASX = ps['vmaxASX'].value
    vmaxfASTA = ps['vmaxfASTA'].value
    vmaxrASTA = ps['vmaxrASTA'].value
    # vmaxAA1 = ps['vmaxAA1'].value
    # vmaxAA2 = ps['vmaxAA2'].value
    vmaxASPT = ps['vmaxASPT'].value
    vmaxACL = ps['vmaxACL'].value
    vmaxgrowth = ps['vmaxgrowth'].value
    K_iLACtoHK = ps['K_iLACtoHK'].value
    # K_iLACtoPFK = ps['K_iLACtoPFK'].value
    K_iLACtoGLNS = ps['K_iLACtoGLNS'].value
    K_iLACtoPYR = ps['K_iLACtoPYR'].value    
    K_iG6P = ps['K_iG6P'].value
    # K_mATPtoADP = ps['K_mATPtoADP'].value
    # K_mADPtoATP = ps['K_mADPtoATP'].value
    # K_mNADHtoNAD = ps['K_mNADHtoNAD'].value
    # K_mNADtoNADH = ps['K_mNADtoNADH'].value
    # K_mNADPtoNADPH = ps['K_mNADPtoNADPH'].value
    # K_iPEP = ps['K_iPEP'].value
    K_iPYR = ps['K_iPYR'].value
    # K_aF6P = ps['K_aF6P'].value
    K_mG6P = ps['K_mG6P'].value
    K_mF6P = ps['K_mF6P'].value
    K_mGAP = ps['K_mGAP'].value
    K_mPEP = ps['K_mPEP'].value
    # K_mPYR = ps['K_mPYR'].value
    K_mLAC = ps['K_mLAC'].value
    K_mEPYR = ps['K_mEPYR'].value
    K_mELAC = ps['K_mELAC'].value
    K_mFUM = ps['K_mFUM'].value
    # K_mEGLU = ps['K_mEGLU'].value
    K_mEASP = ps['K_mEASP'].value
    K_mSUC = ps['K_mSUC'].value
    
    K_mRu5P = ps['K_mRu5P'].value
    K_mPYR = ps['K_mPYR'].value
    K_mAcCoA = ps['K_mAcCoA'].value
    K_mOAA = ps['K_mOAA'].value
    K_mCIT = ps['K_mCIT'].value
    K_mAKG = ps['K_mAKG'].value
    K_mMAL = ps['K_mMAL'].value
    K_mEGLN = ps['K_mEGLN'].value
    K_mNH4 = ps['K_mNH4'].value
    K_mALA = ps['K_mALA'].value
    K_mGLC = ps['K_mGLC'].value
    K_mGLN = ps['K_mGLN'].value
    K_mGLU = ps['K_mGLU'].value
    
    K_aF6P = ps['K_aF6P'].value
    K_aGLN = ps['K_aGLN'].value
    K_mSER = ps['K_mSER'].value
    K_mASP = ps['K_mASP'].value
    K_iGLN = ps['K_iGLN'].value
    
    mu = mu_fit

    
    # AcCoA, AKG, ADP, AMP, ATP, CIT, CoA, Cr, F6P, G6P,GAP ,GLU ,GLY ,MAL ,NAD ,NADH, NADP, NADPH, OAA, O2,PEP,PCr,Pi,ARG, R5P,SUC,X5P,PYR, ALA, ASP, ASX, GLY, HIS, ILE, LUE, LYS, SER, TYR, VAL, GLC, GLN, EGLU, LAC, NH4, BIO, X  = xs
    # AcCoA, AKG, CIT, CO2, F6P, G6P, GAP, GLU, GLY, MAL, OAA, PEP, FUM, Ru5P, SUC, PYR, EPYR, ALA, ASP, EASP, GLY, SER, GLC, GLN, EGLN, EGLU, LAC, ELAC, NH4, BIO, LIPID, X = xs
    AcCoA, AKG, CIT, CO2, F6P, G6P, GAP, GLU, GLY, MAL, OAA, PEP, FUM, Ru5P, SUC, PYR, ALA, ASP, LAC, GLN, EGLY, SER, GLC, EGLN, EGLU, EPYR, EASP, EALA, ELAC, NH4, LIPID, Bio, X = xs

    # GLC, X  = xs
   
    ##Glycolysis
    vHK = vmaxHK * GLC/(K_mGLC+GLC) * K_iLACtoHK/(K_iLACtoHK+ELAC)  * K_iG6P/(K_iG6P+G6P)
    
    # vHK = vmaxHK * GLC/(K_mGLC+GLC) * K_iLACtoHK/(K_iLACtoHK+LAC) #* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) * K_iG6P/(K_iG6P+G6P)
    
    vPGI = vmaxPGI * G6P/(K_mG6P+G6P) #* K_iPEP/(K_iPEP+PEP)
    
    # vPFKALD = vmaxPFKALD * F6P/(K_mF6P+F6P) * K_iG6P/(K_iG6P+G6P) * K_iLACtoPFK/(K_iLACtoPFK+LAC)#* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP))
    
    vPFKALD = vmaxPFKALD * F6P/(K_mF6P+F6P) #* K_iG6P/(K_iG6P+G6P) #* K_iLAC/(K_iLAC+LAC)#* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP))
    
    vPGK = vmaxPGK * GAP/(K_mGAP+GAP) #* (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) #* Pi/(K_mPi+Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP)) 
    
    vPK = vmaxPK * PEP/(K_mPEP*(1 + K_aF6P/F6P)+PEP) #* (ADP/ATP)/(K_mADPtoATP + ADP/ATP)
    
    # vLDH = vmaxfLDH * PYR/(K_mPYR+PYR) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD)) - vmaxrLDH * LAC/(K_mLAC+LAC) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) * K_iPYR/(K_iPYR+PYR)
    
    vLDHf = vmaxfLDH * PYR/(K_mPYR+PYR)
    
    vLDHr = vmaxrLDH * LAC/(K_mLAC+LAC)  * K_iPYR/(K_iPYR+PYR)
    
    vPYRT = vmaxPYRT * EPYR/(K_mEPYR+EPYR) * K_iLACtoPYR/(K_iLACtoPYR+ELAC)
    
    vLACTf = vmaxfLACT * LAC/(K_mLAC+LAC)
    
    vLACTr = vmaxrLACT * ELAC/(K_mELAC+ELAC)
    
    ###PPP
    vOP = vmaxOP * G6P/(K_mG6P+G6P) #* (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))
    
    vNOP = vmaxNOP * Ru5P/(K_mRu5P+Ru5P)
        
    ###TCA
    vPDH = vmaxPDH * PYR/(K_mPYR + PYR) #* (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH))
    
    vCS = vmaxCS * AcCoA/(K_mAcCoA + AcCoA) * OAA/(K_mOAA + OAA)
    
    # vCITSISOD = vmaxfCITSISOD * CIT/(K_mCIT + CIT) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrCITSISOD * AKG/(K_mAKG + AKG) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))
    
    vCITSISODf = vmaxfCITSISOD * CIT/(K_mCIT + CIT)
    
    vCITSISODr = vmaxrCITSISOD * AKG/(K_mAKG + AKG)
    
    vAKGDH = vmaxAKGDH * AKG/(K_mAKG+AKG) #* (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) # * Pi/(K_mPi+Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))
                                                                                                                                                                                                           
    # vSDHFUM = vmaxfSDHFUM * SUC/(K_mSUC + SUC) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrSDHFUM * MAL/(K_mMAL+MAL) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))
    
    vSDH = vmaxSDH * SUC/(K_mSUC + SUC)
    
    vFUMf = vmaxfFUM * FUM/(K_mFUM + FUM) 
    
    vFUMr = vmaxrFUM * MAL/(K_mMAL+MAL)
    
    # vMDH = vmaxfMDH * MAL/(K_mMAL + MAL) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrMDH * OAA/(K_mOAA + OAA) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))
    
    vMDHf = vmaxfMDH * MAL/(K_mMAL + MAL)
    
    vMDHr = vmaxrMDH * OAA/(K_mOAA + OAA)
    
    ##Anaplerosis and Amino Acid
    
    vME = vmaxME * MAL/(K_mMAL + MAL) #* (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH)) 
    
    vPC = vmaxPC * PYR/(K_mPYR + PYR)
    
    # vGLNS = vmaxfGLNS * GLN/(K_mGLN + GLN) * (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) - vmaxrGLNS * GLU/(K_mGLU + GLU) * (ADP/ATP)/(K_mADPtoATP + (ADP/ATP)) * NH4/(K_mNH4 + NH4)
    
    vGLNSf = vmaxfGLNS * GLN/(K_mGLN + GLN) * K_iLACtoGLNS/(K_iLACtoGLNS+ELAC)
    
    vGLNSr = vmaxrGLNS * GLU/(K_mGLU + GLU) * NH4/(K_mNH4 + NH4)
    
    # vGLNS = vmaxfGLNS * GLN/(K_mGLN + GLN) - vmaxrGLNS * GLU/(K_mGLU + GLU) * NH4/(K_mNH4 + NH4)
    
    # vGLDH = vmaxfGLDH * GLU/(K_mGLU + GLU) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) -vmaxrGLDH * AKG/(K_mAKG+AKG) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD)) * NH4/(K_mNH4 + NH4)
    
    vGLDHf = vmaxfGLDH * GLU/(K_mGLU + GLU)
    
    vGLDHr = vmaxrGLDH * AKG/(K_mAKG+AKG) * NH4/(K_mNH4 + NH4)
                                                             
    vAlaTAf = vmaxfAlaTA * GLU/(K_mGLU + GLU) * PYR/(K_mPYR +PYR)
    
    vAlaTAr = vmaxrAlaTA * ALA/(K_mALA + ALA) * AKG/(K_mPYR + AKG) * (1+K_aGLN/GLN)
    
    vAlaT = vmaxAlaT * ALA/(K_mALA + ALA)
    
    # vAlaTA = vmaxfAlaTA * PYR/(K_mPYR +PYR) - vmaxrAlaTA * ALA/(K_mALA + ALA) * (1+K_aGLN/GLN)
    
    vGluT = vmaxGluT * GLU/(K_mGLU + GLU) #* Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP)) 
    
    vGlnT = vmaxGlnT * EGLN/(K_mEGLN + EGLN) * K_iGLN/(K_iGLN+GLN) #* Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP)) 
    
    # vresp = vmaxresp * O2/(K_mO2 + O2) * NADH/(K_mNADH + NADH) #* Pi/(K_mPi + Pi)  #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))
    
    # vleak = vmaxleak * NADH/(K_mNADH + NADH)
    
    # vATPase = vmaxATPase * ATP/(K_mATP + ATP)
    
    # vAK = vmaxfAK * ATP/(K_mATP + ATP) * AMP/(K_mAMP + AMP) - vmaxrAK * ADP/(K_mADP + ADP)
    
    # vCK = vmaxfCK * ADP/(K_mADP + ADP) * PCr/(K_mPCr + PCr) - vmaxrCK * ATP/(K_mATP + ATP) * Cr/(K_mCr + Cr)
    
    # vPPRiBP = vmaxPPRibP * R5P/(K_mR5P + R5P) * ASP/(K_mASP + ASP) * GLN/(K_mGLN + GLN) * GLY/(K_mGLY + GLY)
    
    # vNADPHox = vmaxNADPHox #* NADPH/(K_mNADPH + NADPH)
    
    vSAL = vmaxSAL * SER/(K_mSER + SER)
    
    # vASX = vmaxASX * ASX/(K_mASX + ASX)
    
    vASTAf = vmaxfASTA * ASP/(K_mASP + ASP) * AKG/(K_mAKG + AKG)
    
    vASTAr = vmaxrASTA * GLU/(K_mGLU + GLU) * OAA/(K_mOAA + OAA) * NH4/(K_mNH4 + NH4)
    
    vASPT = vmaxASPT * EASP/(K_mEASP + EASP) #* Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP)) 
    
    # vAA1 = vmaxAA1 * HIS/(K_mHIS + HIS) * ARG/(K_mARG + ARG) * AKG/(K_mAKG + AKG)
    
    # vAA2 = vmaxAA2 * LYS/(K_mLYS + LYS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * VAL/(K_mVAL + VAL) * HIS/(K_mHIS + HIS) * TYR/(K_mTYR + TYR) * AKG/(K_mAKG + AKG) * (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) * (NAD/ NADH)/(K_mNADtoNADH + (NAD/NADH)) * (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))
    
    # vAA2 = vmaxAA2 * LYS/(K_mLYS + LYS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * VAL/(K_mVAL + VAL) * HIS/(K_mHIS + HIS) * TYR/(K_mTYR + TYR) * AKG/(K_mAKG + AKG)  #* (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))
    
    vACL = vmaxACL * CIT/(K_mCIT + CIT) #* Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP)) 
    
    ##Biomass
    
    # vgrowth = vmaxgrowth * R5P/(K_mR5P + R5P) * G6P/(K_mG6P + G6P) * GLN/(K_mGLN + GLN) * ALA/(K_mALA + ALA) * ARG/(K_mARG + ARG) * ASP/(K_mASP + ASP) * HIS/(K_mHIS + HIS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * LYS/(K_mLYS + LYS) * SER/(K_mSER + SER) * TYR/(K_mTYR + TYR) * VAL/(K_mVAL+VAL) * GLY/(K_mGLY + GLY) #* (ATP/ADP)/(K_mATPtoADP+ (ATP/ADP))

    vgrowth = vmaxgrowth * ALA/(K_mALA + ALA) * ASP/(K_mASP + ASP) * GLN/(K_mGLN + GLN) * GLU/(K_mGLU + GLU) * EGLY/(K_mGLY + EGLY) * SER/(K_mSER + SER) * GLC/(K_mGLC+GLC)
    #G6P/(K_mG6P + G6P)   * ARG/(K_mARG + ARG) * HIS/(K_mHIS + HIS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * LYS/(K_mLYS + LYS) * TYR/(K_mTYR + TYR) * VAL/(K_mVAL+VAL) #* (ATP/ADP)/(K_mATPtoADP+ (ATP/ADP))


    # v = [vHK, vPGI, vPFKALD, vPGK, vPK, vLDH, vG6PDHPGLcDH, vEP, vTKTA, vPDH, vCS, vCITSISOD,  
    # vAKGDH, vSDHFUM, vMDH, vME, vPC, vGLNS, vGLDH, vAlaTA, 
    # vGluT, vresp, vleak, vATPase, vAK, vCK, vPPRiBP, vNADPHox, vSAL, vASX, vASTA, vAA1, vAA2, vgrowth]
    
    v= [vHK,	 vPGI, vPFKALD, vPGK	, vPK, vLDHf, vLDHr, vPYRT, vLACTf, vLACTr, vOP, vNOP, vPDH, vCS, vCITSISODf, vCITSISODr, vAKGDH, vSDH,	
        vFUMf, vFUMr, vMDHf, vMDHr, vME, vPC, vGLNSf, vGLNSr, vGLDHf, vGLDHr, vAlaTAf, vAlaTAr, vAlaT, vGluT, vGlnT, vSAL, vASTAf, vASTAr, 	vASPT, vACL, vgrowth]


    du = N @ v * X
    dx = mu*X
       # N[2,] @ v
    if mode == 1:
        return np.append(du, dx).tolist()
    if mode == 0:
        return np.append(v, mu).tolist()
    # return np.hstack([du[:2], np.repeat(0,3), du[-40:], dx]).tolist()

### ADP, AMP, ATP



def g(t, x0, ps, t_eval=None):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    # x = odeint(f, x0, t, args=(ps,))
    start = time.time()
    x = scipy.integrate.solve_ivp(f_m, t,  x0, args=(ps,), t_eval=t_eval, method ='RK45', atol = 1E-3, rtol = 1E-6 )
    end = time.time()
    print('Solving took: ' + str((end-start)) + ' sec')
    return x
        
def residual(ps, ts, data, std, mean):   
# def residual(ps, ts, data, std, flux, flux_std, mean):    
    sum_ls = []
    for i in range(0, len(data)):
        # Calculate Residual about measable state
        for r in range(0, 6):
            batch_data = data[i][(size*r):(size*r+5)]
            # print(batch_data)
            # batch_data = np.vstack((batch_data[:,6],batch_data[:,10],batch_data[:,12], batch_data[:,11]+batch_data[:,12], batch_data[:,13:18].T, batch_data[:,21], batch_data[:,24:26].T, batch_data[:,4], batch_data[:,7:9].T, batch_data[:,5], batch_data[:,9], batch_data[:,26], batch_data[:,3]))
            # batch_data = batch_data.astype(float)
            # x0 = np.hstack([S1_0*1E6*batch_data[-1,0], batch_data[:,0]])
            # x0 = np.hstack([S1_0*1E3*batch_data[0,-1], batch_data[0,:]])
            # model = g([ts[0], ts[-1]], x0, ps, t_eval=ts)
            # sum_ls.append(((model.y[27:] - batch_data.T)/std[i].T).ravel())
            # print(ps)
            x0 = np.hstack([S1_0*1E3*batch_data[0,-1], batch_data[0,:]])
            model = g([ts[0], ts[-1]], x0, ps, t_eval=ts)
            pred = model.y[20:]
            measure_index = np.array([1,2,3,4,5,6,7,8,9,11])
            # print(pred[measure_index])
            # sum_ls.append(((pred[measure_index] - batch_data.T[measure_index])/mean[i].T[measure_index]).ravel())
            sum_ls.append(((pred[measure_index] - batch_data.T[measure_index])).ravel())
            # sum_ls.append(((pred[measure_index] - batch_data.T[measure_index])/std[i].T[measure_index]).ravel())
            # sum_ls.append(((pred[measure_index] - batch_data.T[measure_index])).ravel())
            
        # Calculate Residual about flux rate at 48-th hour
        # x0_case = np.hstack([S1_0*1E3*mean[i][0,-1], mean[i][0,:]])
        # model_case = g([ts[0], ts[-1]], x0_case, ps, t_eval=ts)
        # u_36 = model_case.y[:,-2]   
        # pred_48 = [f_m(0, u_36, ps, 0)[index] for index in mea_flux]    
        # sum_ls.append(((pred_48 - (flux[:,i]/1E3))/(flux_std[:,i]/1E3)).ravel())
            
    residuals = np.concatenate(sum_ls)
    # remove NaN
    residuals = residuals[~np.isnan(residuals)]
    return residuals


    # sum_ls = []
    # # for d in range()
    # # for i in range(18):

    # batch_data = data[(size*i):(size*i+5)]
    # batch_data = np.vstack((batch_data[:,6],batch_data[:,10],batch_data[:,12], batch_data[:,11]+batch_data[:,12], batch_data[:,13:18].T, batch_data[:,21], batch_data[:,24:26].T, batch_data[:,4], batch_data[:,7:9].T, batch_data[:,5], batch_data[:,9], batch_data[:,26], batch_data[:,3]))
    # batch_data = batch_data.astype(float)
    # x0 = np.hstack([S1_0*1000*batch_data[-1,0], batch_data[:,0]])
    # # x0 = np.hstack([batch_data_0[batch_id,4],  batch_data_0[batch_id,3]]).tolist()
    # # xs = np.hstack([batch_data_0[batch_id,4], 5e-08*1000*batch_data[-1,0], batch_data_0[batch_id,3]]).tolist()
    # # x0 = np.hstack([batch_data[-7,0], 5e-08, batch_data[-1,0]]).tolist()
    # # ps = params
    # model = g([ts[0], ts[-1]], x0, ps, t_eval=ts)
    # # f = interp1d(out.t, out.y, kind='quadratic')
    # #
    # # model = f(ts)
    # # g(t,x0,params)
    

    # # residuals = np.concatenate(sum_ls)
    # # remove NaN
    # residuals = (model.y.T[:,24:] - batch_data.T).flatten()
    # residuals = residuals[~np.isnan(residuals)]
    # # print(residuals)
    # return residuals



    
###set parameters incluing bounds
params = Parameters()

params.add('vmaxHK', value=2.90368013, min=0, max=0.04*1E2)
params.add('vmaxPGI', value=1.59542542, min=0, max=0.04*1E2)
params.add('vmaxPFKALD', value=1.57091753, min=0, max=0.04*1E2)
params.add('vmaxPGK', value=3.25750739, min=0, max=0.04*1E2)
params.add('vmaxPK', value=3.27100711, min=0, max=0.04*1E2)
params.add('vmaxfLDH', value=3.28497662, min=0, max=0.1*1E2)
params.add('vmaxrLDH', value=0.052501315, min=0, max=0.04*1E2)
params.add('vmaxPYRT', value=0.17054778, min=0, max=0.04*1E2)
params.add('vmaxfLACT', value=2.95451233, min=0, max=0.04*1E2)
params.add('vmaxrLACT', value=0.52048169, min=0, max=0.04*1E2)

params.add('vmaxOP', value=0.04985896, min=0, max=0.05)
params.add('vmaxNOP', value=0.04613358, min=0, max=0.05)

# params.add('vmaxG6PDHPGLcDH', value=7.88E-05*1E3, min=0, max=0.04*1E2)
# params.add('vmaxEP', value=7.88E-05*1E3, min=0, max=0.04*1E2)
# params.add('vmaxTKTA', value=7.88E-05*1E3, min=0, max=0.04*1E2)
params.add('vmaxPDH', value=0.37899972, min=0, max=0.04*1E2)
params.add('vmaxCS', value=0.37886693, min=0, max=0.04*1E2)
# params.add('vmaxrCS', value=0.11089545, min=0, max=0.04*1E2)
params.add('vmaxfCITSISOD', value=0.40391085, min=0, max=0.04*1E2)
params.add('vmaxrCITSISOD', value=0.10262376, min=0, max=0.04*1E2)
params.add('vmaxAKGDH', value=0.53036265, min=0, max=0.04*1E2)
params.add('vmaxSDH', value=0.23036265, min=0, max=0.04*1E2)
params.add('vmaxfFUM', value=0.23235193, min=0, max=0.04*1E2)
params.add('vmaxrFUM', value=0.08923513, min=0, max=0.04*1E2)
params.add('vmaxfMDH', value=0.23430565, min=0, max=0.04*1E2)
params.add('vmaxrMDH', value=0.08923513, min=0, max=0.04*1E2)
params.add('vmaxME', value=0.20047941, min=0, max=0.04*1E2)
params.add('vmaxPC', value=0.11195132, min=0, max=0.04*1E2)
params.add('vmaxfGLNS', value=1.13622280, min=0, max=0.04*1E2)
params.add('vmaxrGLNS', value=0.28052217, min=0, max=0.04*1E2)
params.add('vmaxfGLDH', value=0.25914301, min=0, max=0.04*1E2)
params.add('vmaxrGLDH', value=0.09426904, min=0, max=0.04*1E2)
params.add('vmaxfAlaTA', value=0.6095075, min=0, max=0.04*1E2)
params.add('vmaxrAlaTA', value=0.05103329, min=0, max=0.04*1E2)
params.add('vmaxAlaT', value=1.93407806, min=0, max=0.04*1E2)
params.add('vmaxGluT', value=0.12855746, min=0, max=0.04*1E2)
params.add('vmaxGlnT', value=1.8119001, min=0, max=0.04*1E2)

# params.add('vmaxresp', value=7.00E-03*1E3, min=0, max=0.03*1E3)
# params.add('vmaxATPase', value=3.60E-03*1E3, min=0, max=0.03*1E3)
# params.add('vmaxleak', value=8.70E-04*1E3, min=0, max=0.03*1E3)
# params.add('vmaxfAK', value=2.30E-03*1E3, min=0, max=0.03*1E3)
# params.add('vmaxrAK', value=9.00E-04*1E3, min=0, max=0.03*1E3)
# params.add('vmaxfCK', value=1.00E-03*1E3, min=0, max=0.03*1E3)
# params.add('vmaxrCK', value=3.00E-04*1E3, min=0, max=0.03*1E3)
# params.add('vmaxPPRibP', value=6.00E-09*1E3, min=0, max=0.03*1E3)
# params.add('vmaxNADPHox', value=4.00E-02*1E3, min=0, max=0.05*1E3)
params.add('vmaxSAL', value=0.01259358, min=0, max=0.04*1E2)
# params.add('vmaxASX', value=4.8648e-06, min=0, max=0.04*1E2)
params.add('vmaxfASTA', value=0.32312476, min=0, max=0.04*1E2)
params.add('vmaxrASTA', value=0.13279262, min=0, max=0.04*1E2)
params.add('vmaxASPT', value=0.02816117, min=0, max=0.04*1E2)
params.add('vmaxACL', value=0.24860839, min=0, max=0.04*1E2)
# params.add('vmaxAA1', value=1.0995e-05, min=0, max=0.04*1E2)
# params.add('vmaxAA2', value=0.02674256, min=0, max=0.04*1E2)
params.add('vmaxgrowth', value=0.0399329474, min=0, max=0.04*1E2)
params.add('K_iLACtoHK', value=34.6391451, min=0, max=0.03*1E4)
# params.add('K_iLACtoPFK', value=5.12E1, min=0, max=0.03*1E4)
params.add('K_iLACtoGLNS', value=17.0024712, min=0, max=0.03*1E4)
params.add('K_iLACtoPYR', value=5.35248851, min=0, max=0.03*1E4)

params.add('K_iG6P', value=0.54937599, min=0, max=0.04*1E2)
params.add('K_iGLN', value=1.50E-02, min=0, max=0.3*1E3)
# params.add('K_mATPtoADP', value=1.00E+00, min=0, max=0.03*1E3)
# params.add('K_mADPtoATP', value=9.00E-06, min=0, max=0.03*1E3)
# params.add('K_mNADHtoNAD', value=3.50E-03, min=0, max=0.03*1E3)
# params.add('K_mNADtoNADH', value=5.00E-01, min=0, max=0.03*1E3)
# params.add('K_mNADPtoNADPH', value=1.00E-03, min=0, max=0.04*1E2)
params.add('K_iPEP', value=5.12E-1, min=0, max=0.3*1E3)
params.add('K_iPYR', value=0.31260487, min=0, max=0.03*1E3)
params.add('K_mEPYR', value=0.18531106, min=0, max=0.3*1E1)
params.add('K_mGLC', value= 1.91892692, min=0, max=0.3*1E1)
params.add('K_mGLN', value=0.26124163, min=0, max=0.3*1E1)
params.add('K_mGLU', value=0.31208884, min=0, max=0.3*1E1)



# params.add('K_aF6P', value=5.12E-1, min=0, max=0.03*1E3)
# K_iPYR = 3.00E-08
# K_aF6P = 1.00E-06
params.add('K_mG6P', value=2.12E-1, min=0, max=0.3*1E1)
params.add('K_mF6P', value=0.18599337, min=0, max=0.3*1E1)
params.add('K_mGAP', value=2.12E-1, min=0, max=0.3*1E1)
params.add('K_mPEP', value=2.12E-1, min=0, max=0.3*1E1)
params.add('K_mFUM', value=1.12E-1, min=0, max=0.3*1E1)
params.add('K_mSUC', value=1.12E-1, min=0, max=0.3*1E1)
params.add('K_mRu5P', value=1.12E-1, min=0, max=0.3*1E1)
params.add('K_mAcCoA', value=0.13298417, min=0, max=0.3*1E1)
params.add('K_mOAA', value=1.12E-1, min=0, max=0.3*1E1)
params.add('K_mCIT', value=2.12E-1, min=0, max=0.3*1E1)
params.add('K_mAKG', value=1.12E-1, min=0, max=0.3*1E1)
params.add('K_mMAL', value=1.12E-1, min=0, max=0.3*1E1)
params.add('K_mALA', value=2.12E-1, min=0, max=0.3*1E1)
params.add('K_mNH4', value=2.12E-1, min=0, max=0.3*1E1)
params.add('K_mELAC', value=5.00E+00, min=0, max=0.3*1E1)
# params.add('K_mEGLU', value=5.00E+00, min=0, max=0.3*1E3)
params.add('K_mEGLN', value=1.00E+00, min=0, max=0.3*1E3)
params.add('K_mEASP', value=0.64521347, min=0, max=0.3*1E3)
params.add('K_mPYR', value=2.12E-1, min=0, max=0.3*1E1)
params.add('K_mLAC', value=0.04256268, min=0, max=0.3*1E3)

params.add('K_aF6P', value=1.00E-06, min=0, max=0.3*1E3)
params.add('K_aGLN', value=1.00E+01, min=0, max=0.3*1E3)
params.add('K_mSER', value=1.50E-02, min=0, max=0.3*1E3)
params.add('K_mASP', value=3.31472182, min=0, max=0.3*1E3)



# meta = {0:	'AcCoA', 1:	 'AKG', 2: 'ADP', 3: 'AMP', 4: 'ATP', 5:	 'CIT', 6: 'CoA',7:	 'Cr', 8:	 'F6P',9:	 'G6P',10:	 'GAP' ,11:	 'GLU' ,12:	 'GLY' 
# ,13:	 'MAL' ,14:	 'NAD' ,15:	 'NADH',16:	 'NADP',17:	 'NADPH',18:	 'OAA',19:	 'O2',20:	 'PEP',21:	 'PCr'
# ,22:	 'Pi',23:	 'ARG',24:	 'R5P',25:	 'SUC',26:	 'X5P',27:	 'PYR',28:	 'ALA',29:	 'ASP',30:	 'ASX'
# ,31:	 'GLY',32:	 'HIS',33:	 'ILE',34:	 'LUE',35:	 'LYS',36:	 'SER',37:	 'TYR',38:	 'VAL',39:	 'GLC'
# ,40:	 'GLN',41:	 'EGLU',42:	 'LAC',43:	 'NH4',44:	 'BIO',45:	 'X'}


meta = {0:	'AcCoA', 1:	 'AKG', 2: 'CIT', 3: 'CO2', 4: 'F6P', 5:	 'G6P', 6: 'GAP',7:	 'GLU', 8:	 'GLY',9:	 'MAL',10:	 'OAA' ,11:	 'PEP' ,12:	 'FUM' 
,13:	 'Ru5P' ,14:	 'SUC' ,15:	 'PYR',16:	 'ALA',17:	 'ASP',18:	 'LAC',19:	 'GLN',20:	 'EGLY',21:	 'SER'
,22:	 'GLC',23:	 'EGLN',24:	 'EGLU',25:	 'EPYR',26:	 'EASP',27:	 'EALA',28:	 'ELAC',29:	 'NH4',30:	 'LIPID'
,31:	 'BIO',32:	 'X'}

# meta_index = np.array([28, 29 ,36, 39, 40, 41, 42, 43, 45])
meta_index = np.array([21, 22, 23 ,24, 25, 26, 27, 28, 29, 32])


dataset = {0:'HGHL', 1:'HGLL', 3:'LGHL', 2:'LGLL'}



#############################
# Train 0:HGHL, 1:HGLL, 3:LGHL, 2:LGLL
# Test 

indices = [0,1,2,3]
data_train = [data_whole[index] for index in indices]
std_train = [std_whole[index] for index in indices]
# flux_train = flux_48[:,indices]
# flux_std_train = flux_std_48[:,indices]
mean_train = [mean_whole[index] for index in indices]

# result_all = minimize(residual, params, args=(t, data_train, std_train, mean_train ), method='leastsq')
# res_all = fit_report(result_all)

# Load data (deserialize)
with open('result_all.pickle', 'rb') as handle:
    result_all = pickle.load(handle)

parameters = result_all.params


test_index = 3

#ALL
x0_ALL = np.hstack([S1_0*1000* mean_whole[test_index][0,-1], mean_whole[test_index][0]])
pred_ALL = g([t[0], t[-1]], x0_ALL, parameters, t).y  # g(t, x0_LGHL, parameters)
pred_ALL.T

t_pred = np.linspace(0, 48,100)
pred = g([t[0], t[-1]], x0_ALL, parameters, t_pred).y 

for i in meta_index:
    fig, ax = plt.subplots()
    
    ax.errorbar([0,12,24,36,48],mean_whole[test_index][:,(i-20)], yerr = 1.96*std_whole[test_index][:,(i-20)], linestyle='None', marker = 'o', color='blue', label='Measurement', )
    ax.plot(t_pred,pred[i], color='black', label='Prediction')
    ax.set_title(str(meta[i]) + '(' + dataset[test_index] + ')',fontsize=15)
    ax.set_xlabel('Time (h)',fontsize=15)
    if str(meta[i]) == 'X':
        ax.set_ylabel('10^6 cells/mL',fontsize=15)
    else:      
        ax.set_ylabel('Conc. (mM)',fontsize=15)
    if np.max(mean_whole[test_index][:,(i-20)]) < 0.5:
        ax.set_ylim(0,0.5)
    elif np.max(mean_whole[test_index][:,(i-20)]) < 1:
        ax.set_ylim(0,1)
    elif np.max(mean_whole[test_index][:,(i-20)]) < 5:
        ax.set_ylim(0,5)
    else:
        ax.set_ylim(0,40)
    plt.xticks([0,12,24,36,48],fontsize=12)
    leg = ax.legend(fontsize=15)
    plt.savefig(str(meta[i]) + '(' + dataset[test_index] + ')' + '.svg')
    plt.show()


dudt = np.zeros([len(t_pred), 40])
for i in range(0, len(t_pred)):
    dudt[i,] = f_m(t, pred.T[i,], parameters,0)    
dudt = pd.DataFrame(dudt)
dudt.columns = flux_list
dudt.to_csv('prediction_flux_ALL_cross.csv')

pred = pd.DataFrame(pred.T)
pred.columns = meta_list
pred.to_csv('prediction_ALL_cross.csv')





    