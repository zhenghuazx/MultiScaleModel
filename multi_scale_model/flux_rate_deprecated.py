import numpy as np
import scipy

mu = 0.04531871789155581
K_mGLC = 4.19E+00
K_mGLN = 1.58E+00
K_mLAC = 5.00E+00
K_mNH4 = 5.00E-01
K_mALA = 1.00E+00
K_mARG = 5.00E-02
K_mASP = 1.50E-02
K_mASX = 1.50E-01
K_mGLY = 5.00E-02
K_mHIS = 5.00E-01
K_mILE = 2.50E-02
K_mLUE = 2.50E-02
K_mLYS = 5.00E-02
K_mSER = 1.50E-02
K_mTYR = 5.00E-02
K_mVAL = 5.00E-02
K_aGLN = 1.00E+01
K_mGLNmAb = 1.00E-02
K_mG6P = 8.20E-08
K_mF6P = 4.00E-07
K_mGAP = 8.00E-07
K_mPEP = 1.70E-07
K_mR5P = 1.30E-08
K_mX5P = 3.40E-08
K_mPYR = 9.80E-07
K_mAcCoA = 9.60E-07
K_mOXA = 2.40E-07
K_mCIT = 1.00E-07
K_mAKG = 8.60E-07
K_mSUC = 2.80E-08
K_mMAL = 8.50E-07
K_mGLU = 1.28E-04
K_mATP = 4.20E-06
K_mADP = 3.65E-07
K_mAMP = 2.52E-08
K_mPCr = 7.70E-06
K_mCr = 6.00E-07
K_mO2 = 4.00E-06
K_mPi = 1.00E-06
K_mNADH = 1.00E-07
K_mNADPH = 1.00E-09
K_mATPtoADP = 1.00E+00
K_mADPtoATP = 9.00E-06
K_mNADHtoNAD = 3.50E-03
K_mNADtoNADH = 5.00E-01
K_mNADPtoNADPH = 1.00E-03
# K_iG6P = 5.12E-08
K_iPEP = 2.00E-07
K_iPYR = 3.00E-08
K_aF6P = 1.00E-06
vmaxresp = 7.00E-03 * 1E3
vmaxATPase = 3.60E-03 * 1E3
vmaxleak = 8.70E-04 * 1E3
vmaxfAK = 2.30E-03 * 1E3
vmaxrAK = 9.00E-04 * 1E3
vmaxfCK = 1.00E-03 * 1E3
vmaxrCK = 3.00E-04 * 1E3
vmaxPPRibP = 6.00E-09 * 1E3
vmaxNADPHox = 4.00E-02 * 1E3
vmaxG6PDHPGLcDH = 7.88E-05 * 1E3
vmaxEP = 7.88E-05 * 1E3
vmaxTKTA = 7.88E-05 * 1E3


S1_0 = np.array([1.40E-07,
                 4.00E-07,
                 2.00E-07,
                 2.00E-08 * 1E10,
                 4.00E-06,
                 3.00E-07,
                 5.00E-08,
                 4.00E-07,
                 7.00E-04, 8.00E-06, 1E-6, 9.00E-07, 2.00E-08, 9.00E-13, 5.30E-12, 5.00E-05, 8.00E-06,
4.00E-08, 2.00E-06, 2.00E-06, 2.8, 1.50E-07, 5.30E-07, 9.20E-08])

def f_m(t, xs, ps, N, mode=1):
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
    # vmaxG6PDHPGLcDH = ps['vmaxG6PDHPGLcDH'].value
    # vmaxEP = ps['vmaxEP'].value
    # vmaxTKTA = ps['vmaxTKTA'].value
    vmaxPDH = ps['vmaxPDH'].value
    vmaxfCS = ps['vmaxfCS'].value
    vmaxrCS = ps['vmaxrCS'].value
    vmaxfCITSISOD = ps['vmaxfCITSISOD'].value
    vmaxrCITSISOD = ps['vmaxrCITSISOD'].value
    vmaxAKGDH = ps['vmaxAKGDH'].value
    vmaxfSDHFUM = ps['vmaxfSDHFUM'].value
    vmaxrSDHFUM = ps['vmaxrSDHFUM'].value
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
    vmaxGluT = ps['vmaxGluT'].value
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
    vmaxASX = ps['vmaxASX'].value
    vmaxfASTA = ps['vmaxfASTA'].value
    vmaxrASTA = ps['vmaxrASTA'].value
    vmaxAA1 = ps['vmaxAA1'].value
    vmaxAA2 = ps['vmaxAA2'].value
    vmaxgrowth = ps['vmaxgrowth'].value
    K_iLACtoHK = ps['K_iLACtoHK'].value
    # K_iLACtoPFK = ps['K_iLACtoPFK'].value
    K_iLACtoGLNS = ps['K_iLACtoGLNS'].value
    K_iG6P = ps['K_iG6P'].value
    # K_mATPtoADP = ps['K_mATPtoADP'].value
    # K_mADPtoATP = ps['K_mADPtoATP'].value
    # K_mNADHtoNAD = ps['K_mNADHtoNAD'].value
    # K_mNADtoNADH = ps['K_mNADtoNADH'].value
    # K_mNADPtoNADPH = ps['K_mNADPtoNADPH'].value
    # K_iPEP = ps['K_iPEP'].value
    # K_iPYR = ps['K_iPYR'].value
    # K_aF6P = ps['K_aF6P'].value
    xs[xs < 0] = 0
    AcCoA, AKG, ADP, AMP, ATP, CIT, CoA, Cr, F6P, G6P, GAP, GLU, GLY, MAL, NAD, NADH, NADP, NADPH, OAA, O2, PEP, PCr, Pi, ARG, R5P, SUC, X5P, PYR, ALA, ASP, ASX, GLY, HIS, ILE, LUE, LYS, SER, TYR, VAL, GLC, GLN, EGLU, LAC, NH4, BIO, X = xs
    # GLC, X  = xs

    ##Glycolysis
    vHK = vmaxHK * GLC / (K_mGLC + GLC) * K_iLACtoHK / (K_iLACtoHK + LAC) * K_iG6P / (K_iG6P + G6P)

    # vHK = vmaxHK * GLC/(K_mGLC+GLC) * K_iLACtoHK/(K_iLACtoHK+LAC) #* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) * K_iG6P/(K_iG6P+G6P)

    vPGI = vmaxPGI * G6P / (K_mG6P + G6P)  # * K_iPEP/(K_iPEP+PEP)

    # vPFKALD = vmaxPFKALD * F6P/(K_mF6P+F6P) * K_iG6P/(K_iG6P+G6P) * K_iLACtoPFK/(K_iLACtoPFK+LAC)#* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP))

    vPFKALD = vmaxPFKALD * F6P / (
                K_mF6P + F6P)  # * K_iG6P/(K_iG6P+G6P) #* K_iLAC/(K_iLAC+LAC)#* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP))

    vPGK = vmaxPGK * GAP / (
                K_mGAP + GAP)  # * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) #* Pi/(K_mPi+Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    vPK = vmaxPK * PEP / (K_mPEP * (1 + K_aF6P / F6P) + PEP)  # * (ADP/ATP)/(K_mADPtoATP + ADP/ATP)

    # vLDH = vmaxfLDH * PYR/(K_mPYR+PYR) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD)) - vmaxrLDH * LAC/(K_mLAC+LAC) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) * K_iPYR/(K_iPYR+PYR)

    vLDH = vmaxfLDH * PYR / (K_mPYR + PYR) - vmaxrLDH * LAC / (K_mLAC + LAC) * K_iPYR / (K_iPYR + PYR)

    ###PPP
    vG6PDHPGLcDH = vmaxG6PDHPGLcDH * G6P / (K_mG6P + G6P)  # * (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))

    vEP = vmaxEP * R5P / (K_mR5P + R5P)

    vTKTA = vmaxTKTA * R5P / (K_mR5P + R5P) * X5P / (K_mX5P + X5P)

    ###TCA
    vPDH = vmaxPDH * PYR / (K_mPYR + PYR)  # * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH))

    vCS = vmaxfCS * AcCoA / (K_mAcCoA + AcCoA) * OAA / (K_mOXA + OAA) - vmaxrCS * CIT / (K_mCIT + CIT)

    # vCITSISOD = vmaxfCITSISOD * CIT/(K_mCIT + CIT) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrCITSISOD * AKG/(K_mAKG + AKG) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))

    vCITSISOD = vmaxfCITSISOD * CIT / (K_mCIT + CIT) - vmaxrCITSISOD * AKG / (K_mAKG + AKG)

    vAKGDH = vmaxAKGDH * AKG / (
                K_mAKG + AKG)  # * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) # * Pi/(K_mPi+Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    # vSDHFUM = vmaxfSDHFUM * SUC/(K_mSUC + SUC) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrSDHFUM * MAL/(K_mMAL+MAL) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))

    vSDHFUM = vmaxfSDHFUM * SUC / (K_mSUC + SUC) - vmaxrSDHFUM * MAL / (K_mMAL + MAL)

    # vMDH = vmaxfMDH * MAL/(K_mMAL + MAL) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrMDH * OAA/(K_mOXA + OAA) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))

    vMDH = vmaxfMDH * MAL / (K_mMAL + MAL) - vmaxrMDH * OAA / (K_mOXA + OAA)

    ##Anaplerosis and Amino Acid

    vME = vmaxME * MAL / (K_mMAL + MAL)  # * (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))

    vPC = vmaxPC * PYR / (K_mPYR + PYR)

    # vGLNS = vmaxfGLNS * GLN/(K_mGLN + GLN) * (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) - vmaxrGLNS * GLU/(K_mGLU + GLU) * (ADP/ATP)/(K_mADPtoATP + (ADP/ATP)) * NH4/(K_mNH4 + NH4)

    vGLNS = vmaxfGLNS * GLN / (K_mGLN + GLN) * K_iLACtoGLNS / (K_iLACtoGLNS + LAC) - vmaxrGLNS * GLU / (
                K_mGLU + GLU) * NH4 / (K_mNH4 + NH4)

    # vGLNS = vmaxfGLNS * GLN/(K_mGLN + GLN) - vmaxrGLNS * GLU/(K_mGLU + GLU) * NH4/(K_mNH4 + NH4)

    # vGLDH = vmaxfGLDH * GLU/(K_mGLU + GLU) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) -vmaxrGLDH * AKG/(K_mAKG+AKG) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD)) * NH4/(K_mNH4 + NH4)

    vGLDH = vmaxfGLDH * GLU / (K_mGLU + GLU) - vmaxrGLDH * AKG / (K_mAKG + AKG) * NH4 / (K_mNH4 + NH4)

    vAlaTA = vmaxfAlaTA * GLU / (K_mGLU + GLU) * PYR / (K_mPYR + PYR) - vmaxrAlaTA * ALA / (K_mALA + ALA) * AKG / (
                K_mPYR + AKG) * (1 + K_aGLN / max(GLN, 1e-6))

    # vAlaTA = vmaxfAlaTA * PYR/(K_mPYR +PYR) - vmaxrAlaTA * ALA/(K_mALA + ALA) * (1+K_aGLN/GLN)

    vGluT = vmaxGluT * GLU / (K_mGLU + GLU)  # * Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    vresp = vmaxresp * O2 / (K_mO2 + O2) * NADH / (
                K_mNADH + NADH)  # * Pi/(K_mPi + Pi)  #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    vleak = vmaxleak * NADH / (K_mNADH + NADH)

    vATPase = vmaxATPase * ATP / (K_mATP + ATP)

    vAK = vmaxfAK * ATP / (K_mATP + ATP) * AMP / (K_mAMP + AMP) - vmaxrAK * ADP / (K_mADP + ADP)

    vCK = vmaxfCK * ADP / (K_mADP + ADP) * PCr / (K_mPCr + PCr) - vmaxrCK * ATP / (K_mATP + ATP) * Cr / (K_mCr + Cr)

    vPPRiBP = vmaxPPRibP * R5P / (K_mR5P + R5P) * ASP / (K_mASP + ASP) * GLN / (K_mGLN + GLN) * GLY / (K_mGLY + GLY)

    vNADPHox = vmaxNADPHox  # * NADPH/(K_mNADPH + NADPH)

    vSAL = vmaxSAL * SER / (K_mSER + SER)

    vASX = vmaxASX * ASX / (K_mASX + ASX)

    vASTA = vmaxfASTA * ASP / (K_mASP + ASP) * AKG / (K_mAKG + AKG) - vmaxrASTA * GLU / (K_mGLU + GLU) * OAA / (
                K_mOXA + OAA) * NH4 / (K_mNH4 + NH4)

    vAA1 = vmaxAA1 * HIS / (K_mHIS + HIS) * ARG / (K_mARG + ARG) * AKG / (K_mAKG + AKG)

    # vAA2 = vmaxAA2 * LYS/(K_mLYS + LYS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * VAL/(K_mVAL + VAL) * HIS/(K_mHIS + HIS) * TYR/(K_mTYR + TYR) * AKG/(K_mAKG + AKG) * (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) * (NAD/ NADH)/(K_mNADtoNADH + (NAD/NADH)) * (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))

    vAA2 = vmaxAA2 * LYS / (K_mLYS + LYS) * ILE / (K_mILE + ILE) * LUE / (K_mLUE + LUE) * VAL / (K_mVAL + VAL) * HIS / (
                K_mHIS + HIS) * TYR / (K_mTYR + TYR) * AKG / (
                       K_mAKG + AKG)  # * (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))

    ##Biomass

    # vgrowth = vmaxgrowth * R5P/(K_mR5P + R5P) * G6P/(K_mG6P + G6P) * GLN/(K_mGLN + GLN) * ALA/(K_mALA + ALA) * ARG/(K_mARG + ARG) * ASP/(K_mASP + ASP) * HIS/(K_mHIS + HIS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * LYS/(K_mLYS + LYS) * SER/(K_mSER + SER) * TYR/(K_mTYR + TYR) * VAL/(K_mVAL+VAL) * GLY/(K_mGLY + GLY) #* (ATP/ADP)/(K_mATPtoADP+ (ATP/ADP))

    vgrowth = vmaxgrowth * ALA / (K_mALA + ALA) * ASP / (K_mASP + ASP) * GLN / (K_mGLN + GLN) * GLU / (
                K_mGLU + GLU) * GLY / (K_mGLY + GLY) * SER / (K_mSER + SER) * GLC / (K_mGLC + GLC)
    # G6P/(K_mG6P + G6P)   * ARG/(K_mARG + ARG) * HIS/(K_mHIS + HIS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * LYS/(K_mLYS + LYS) * TYR/(K_mTYR + TYR) * VAL/(K_mVAL+VAL) #* (ATP/ADP)/(K_mATPtoADP+ (ATP/ADP))

    v = [vHK, vPGI, vPFKALD, vPGK, vPK, vLDH, vG6PDHPGLcDH, vEP, vTKTA, vPDH, vCS, vCITSISOD,
         vAKGDH, vSDHFUM, vMDH, vME, vPC, vGLNS, vGLDH, vAlaTA,
         vGluT, vresp, vleak, vATPase, vAK, vCK, vPPRiBP, vNADPHox, vSAL, vASX, vASTA, vAA1, vAA2, vgrowth]

    du = N @ v * X
    dx = mu * X
    # N[2,] @ v
    if mode == 1:
        return np.append(du, dx).tolist()
    if mode == 0:
        return np.append(v, mu).tolist()

def g(t, x0, ps, N, t_eval=None):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    # x = odeint(f, x0, t, args=(ps,))
    x = scipy.integrate.solve_ivp(f_m, t, x0, args=(ps, N, ), t_eval=t_eval, method='RK45', atol=1E-3, rtol=1E-5)
    return x