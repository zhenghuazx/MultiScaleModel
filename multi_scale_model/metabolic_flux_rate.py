import numpy as np
import scipy

K_mGLY = 5.00E-02
mu = 0.04531871789155581


def f_m(t, xs, ps, N, mode=1):
    """
    Bio_Kinetic Model.
    mode: 1: Return du of each metabolite
          0: Return the flux rate of each reaction
    """
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
    xs[xs < 0] = 1e-11
    # AcCoA, AKG, ADP, AMP, ATP, CIT, CoA, Cr, F6P, G6P,GAP ,GLU ,GLY ,MAL ,NAD ,NADH, NADP, NADPH, OAA, O2,PEP,PCr,Pi,ARG, R5P,SUC,X5P,PYR, ALA, ASP, ASX, GLY, HIS, ILE, LUE, LYS, SER, TYR, VAL, GLC, GLN, EGLU, LAC, NH4, BIO, X  = xs
    # AcCoA, AKG, CIT, CO2, F6P, G6P, GAP, GLU, GLY, MAL, OAA, PEP, FUM, Ru5P, SUC, PYR, EPYR, ALA, ASP, EASP, GLY, SER, GLC, GLN, EGLN, EGLU, LAC, ELAC, NH4, BIO, LIPID, X = xs
    AcCoA, AKG, CIT, CO2, F6P, G6P, GAP, GLU, GLY, MAL, OAA, PEP, FUM, Ru5P, SUC, PYR, ALA, ASP, LAC, GLN, EGLY, SER, GLC, EGLN, EGLU, EPYR, EASP, EALA, ELAC, NH4, LIPID, Bio, X = xs

    # GLC, X  = xs

    ##Glycolysis
    vHK = vmaxHK * GLC / (K_mGLC + GLC) * K_iLACtoHK / (K_iLACtoHK + ELAC) * K_iG6P / (K_iG6P + G6P)

    # vHK = vmaxHK * GLC/(K_mGLC+GLC) * K_iLACtoHK/(K_iLACtoHK+LAC) #* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) * K_iG6P/(K_iG6P+G6P)

    vPGI = vmaxPGI * G6P / (K_mG6P + G6P)  # * K_iPEP/(K_iPEP+PEP)

    # vPFKALD = vmaxPFKALD * F6P/(K_mF6P+F6P) * K_iG6P/(K_iG6P+G6P) * K_iLACtoPFK/(K_iLACtoPFK+LAC)#* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP))

    vPFKALD = vmaxPFKALD * F6P / (
                K_mF6P + F6P)  # * K_iG6P/(K_iG6P+G6P) #* K_iLAC/(K_iLAC+LAC)#* (ATP/ADP)/(K_mATPtoADP + (ATP/ADP))

    vPGK = vmaxPGK * GAP / (
                K_mGAP + GAP)  # * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) #* Pi/(K_mPi+Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    vPK = vmaxPK * PEP / (K_mPEP * (1 + K_aF6P / F6P) + PEP)  # * (ADP/ATP)/(K_mADPtoATP + ADP/ATP)

    # vLDH = vmaxfLDH * PYR/(K_mPYR+PYR) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD)) - vmaxrLDH * LAC/(K_mLAC+LAC) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) * K_iPYR/(K_iPYR+PYR)

    vLDHf = vmaxfLDH * PYR / (K_mPYR + PYR)

    vLDHr = vmaxrLDH * LAC / (K_mLAC + LAC) * K_iPYR / (K_iPYR + PYR)

    vPYRT = vmaxPYRT * EPYR / (K_mEPYR + EPYR) * K_iLACtoPYR / (K_iLACtoPYR + ELAC)

    vLACTf = vmaxfLACT * LAC / (K_mLAC + LAC)

    vLACTr = vmaxrLACT * ELAC / (K_mELAC + ELAC)

    ###PPP
    vOP = vmaxOP * G6P / (K_mG6P + G6P)  # * (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))

    vNOP = vmaxNOP * Ru5P / (K_mRu5P + Ru5P)

    ###TCA
    vPDH = vmaxPDH * PYR / (K_mPYR + PYR)  # * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH))

    vCS = vmaxCS * AcCoA / (K_mAcCoA + AcCoA) * OAA / (K_mOAA + OAA)

    # vCITSISOD = vmaxfCITSISOD * CIT/(K_mCIT + CIT) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrCITSISOD * AKG/(K_mAKG + AKG) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))

    vCITSISODf = vmaxfCITSISOD * CIT / (K_mCIT + CIT)

    vCITSISODr = vmaxrCITSISOD * AKG / (K_mAKG + AKG)

    vAKGDH = vmaxAKGDH * AKG / (
                K_mAKG + AKG)  # * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) # * Pi/(K_mPi+Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    # vSDHFUM = vmaxfSDHFUM * SUC/(K_mSUC + SUC) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrSDHFUM * MAL/(K_mMAL+MAL) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))

    vSDH = vmaxSDH * SUC / (K_mSUC + SUC)

    vFUMf = vmaxfFUM * FUM / (K_mFUM + FUM)

    vFUMr = vmaxrFUM * MAL / (K_mMAL + MAL)

    # vMDH = vmaxfMDH * MAL/(K_mMAL + MAL) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) - vmaxrMDH * OAA/(K_mOAA + OAA) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD))

    vMDHf = vmaxfMDH * MAL / (K_mMAL + MAL)

    vMDHr = vmaxrMDH * OAA / (K_mOAA + OAA)

    ##Anaplerosis and Amino Acid

    vME = vmaxME * MAL / (K_mMAL + MAL)  # * (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))

    vPC = vmaxPC * PYR / (K_mPYR + PYR)

    # vGLNS = vmaxfGLNS * GLN/(K_mGLN + GLN) * (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) - vmaxrGLNS * GLU/(K_mGLU + GLU) * (ADP/ATP)/(K_mADPtoATP + (ADP/ATP)) * NH4/(K_mNH4 + NH4)

    vGLNSf = vmaxfGLNS * GLN / (K_mGLN + GLN) * K_iLACtoGLNS / (K_iLACtoGLNS + ELAC)

    vGLNSr = vmaxrGLNS * GLU / (K_mGLU + GLU) * NH4 / (K_mNH4 + NH4)

    # vGLNS = vmaxfGLNS * GLN/(K_mGLN + GLN) - vmaxrGLNS * GLU/(K_mGLU + GLU) * NH4/(K_mNH4 + NH4)

    # vGLDH = vmaxfGLDH * GLU/(K_mGLU + GLU) * (NAD/NADH)/(K_mNADtoNADH + (NAD/NADH)) -vmaxrGLDH * AKG/(K_mAKG+AKG) * (NADH/NAD)/(K_mNADHtoNAD + (NADH/NAD)) * NH4/(K_mNH4 + NH4)

    vGLDHf = vmaxfGLDH * GLU / (K_mGLU + GLU)

    vGLDHr = vmaxrGLDH * AKG / (K_mAKG + AKG) * NH4 / (K_mNH4 + NH4)

    vAlaTAf = vmaxfAlaTA * GLU / (K_mGLU + GLU) * PYR / (K_mPYR + PYR)

    vAlaTAr = vmaxrAlaTA * ALA / (K_mALA + ALA) * AKG / (K_mPYR + AKG) * (1 + K_aGLN / GLN)

    vAlaT = vmaxAlaT * ALA / (K_mALA + ALA)

    # vAlaTA = vmaxfAlaTA * PYR/(K_mPYR +PYR) - vmaxrAlaTA * ALA/(K_mALA + ALA) * (1+K_aGLN/GLN)

    vGluT = vmaxGluT * GLU / (K_mGLU + GLU)  # * Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    vGlnT = vmaxGlnT * EGLN / (K_mEGLN + EGLN) * K_iGLN / (
                K_iGLN + GLN)  # * Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    # vresp = vmaxresp * O2/(K_mO2 + O2) * NADH/(K_mNADH + NADH) #* Pi/(K_mPi + Pi)  #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    # vleak = vmaxleak * NADH/(K_mNADH + NADH)

    # vATPase = vmaxATPase * ATP/(K_mATP + ATP)

    # vAK = vmaxfAK * ATP/(K_mATP + ATP) * AMP/(K_mAMP + AMP) - vmaxrAK * ADP/(K_mADP + ADP)

    # vCK = vmaxfCK * ADP/(K_mADP + ADP) * PCr/(K_mPCr + PCr) - vmaxrCK * ATP/(K_mATP + ATP) * Cr/(K_mCr + Cr)

    # vPPRiBP = vmaxPPRibP * R5P/(K_mR5P + R5P) * ASP/(K_mASP + ASP) * GLN/(K_mGLN + GLN) * GLY/(K_mGLY + GLY)

    # vNADPHox = vmaxNADPHox #* NADPH/(K_mNADPH + NADPH)

    vSAL = vmaxSAL * SER / (K_mSER + SER)

    # vASX = vmaxASX * ASX/(K_mASX + ASX)

    vASTAf = vmaxfASTA * ASP / (K_mASP + ASP) * AKG / (K_mAKG + AKG)

    vASTAr = vmaxrASTA * GLU / (K_mGLU + GLU) * OAA / (K_mOAA + OAA) * NH4 / (K_mNH4 + NH4)

    vASPT = vmaxASPT * EASP / (K_mEASP + EASP)  # * Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    # vAA1 = vmaxAA1 * HIS/(K_mHIS + HIS) * ARG/(K_mARG + ARG) * AKG/(K_mAKG + AKG)

    # vAA2 = vmaxAA2 * LYS/(K_mLYS + LYS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * VAL/(K_mVAL + VAL) * HIS/(K_mHIS + HIS) * TYR/(K_mTYR + TYR) * AKG/(K_mAKG + AKG) * (ATP/ADP)/(K_mATPtoADP + (ATP/ADP)) * (NAD/ NADH)/(K_mNADtoNADH + (NAD/NADH)) * (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))

    # vAA2 = vmaxAA2 * LYS/(K_mLYS + LYS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * VAL/(K_mVAL + VAL) * HIS/(K_mHIS + HIS) * TYR/(K_mTYR + TYR) * AKG/(K_mAKG + AKG)  #* (NADP/NADPH)/(K_mNADPtoNADPH + (NADP/NADPH))

    vACL = vmaxACL * CIT / (K_mCIT + CIT)  # * Pi/(K_mPi + Pi) #* (ADP/ATP)/(K_mADPtoATP + (ADP/ATP))

    ##Biomass

    # vgrowth = vmaxgrowth * R5P/(K_mR5P + R5P) * G6P/(K_mG6P + G6P) * GLN/(K_mGLN + GLN) * ALA/(K_mALA + ALA) * ARG/(K_mARG + ARG) * ASP/(K_mASP + ASP) * HIS/(K_mHIS + HIS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * LYS/(K_mLYS + LYS) * SER/(K_mSER + SER) * TYR/(K_mTYR + TYR) * VAL/(K_mVAL+VAL) * GLY/(K_mGLY + GLY) #* (ATP/ADP)/(K_mATPtoADP+ (ATP/ADP))

    vgrowth = vmaxgrowth * ALA / (K_mALA + ALA) * ASP / (K_mASP + ASP) * GLN / (K_mGLN + GLN) * GLU / (
                K_mGLU + GLU) * EGLY / (K_mGLY + EGLY) * SER / (K_mSER + SER) * GLC / (K_mGLC + GLC)
    # G6P/(K_mG6P + G6P)   * ARG/(K_mARG + ARG) * HIS/(K_mHIS + HIS) * ILE/(K_mILE + ILE) * LUE/(K_mLUE + LUE) * LYS/(K_mLYS + LYS) * TYR/(K_mTYR + TYR) * VAL/(K_mVAL+VAL) #* (ATP/ADP)/(K_mATPtoADP+ (ATP/ADP))

    # v = [vHK, vPGI, vPFKALD, vPGK, vPK, vLDH, vG6PDHPGLcDH, vEP, vTKTA, vPDH, vCS, vCITSISOD,
    # vAKGDH, vSDHFUM, vMDH, vME, vPC, vGLNS, vGLDH, vAlaTA,
    # vGluT, vresp, vleak, vATPase, vAK, vCK, vPPRiBP, vNADPHox, vSAL, vASX, vASTA, vAA1, vAA2, vgrowth]

    v = [vHK, vPGI, vPFKALD, vPGK, vPK, vLDHf, vLDHr, vPYRT, vLACTf, vLACTr, vOP, vNOP, vPDH, vCS, vCITSISODf,
         vCITSISODr, vAKGDH, vSDH,
         vFUMf, vFUMr, vMDHf, vMDHr, vME, vPC, vGLNSf, vGLNSr, vGLDHf, vGLDHr, vAlaTAf, vAlaTAr, vAlaT, vGluT, vGlnT,
         vSAL, vASTAf, vASTAr, vASPT, vACL, vgrowth]

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

