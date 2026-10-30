# Direct MacroModelling-style translation of the linear Smets-Wouters (2007)
# Dynare replication model used by Farkas and Tatar (2023). Shock standard
# deviations are explicit coefficients because the Python state-space runtime
# uses unit-variance structural innovations.
@model Smets_Wouters_2007_linear begin
    # Flexible-price economy
    a[0] = calfa * rkf[0] + (1 - calfa) * wf[0]
    zcapf[0] = rkf[0] / cZcap
    rkf[0] = wf[0] + labf[0] - kf[0]
    kf[0] = kpf[-1] + zcapf[0]
    invef[0] = (invef[-1] + cbetabar * cgamma * invef[1] + pkf[0] / (cgamma ^ 2 * csadjcost)) / (1 + cbetabar * cgamma) + qs[0]
    pkf[0] = -rrf[0] + b[0] / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) + crk / (crk + 1 - ctou) * rkf[1] + (1 - ctou) / (crk + 1 - ctou) * pkf[1]
    cf[0] = (chabb / cgamma) / (1 + chabb / cgamma) * cf[-1] + cf[1] / (1 + chabb / cgamma) + (csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma)) * (labf[0] - labf[1]) - (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)) * rrf[0] + b[0]
    yf[0] = ccy * cf[0] + ciy * invef[0] + g[0] + crkky * zcapf[0]
    yf[0] = cfc * (calfa * kf[0] + (1 - calfa) * labf[0] + a[0])
    wf[0] = csigl * labf[0] + cf[0] / (1 - chabb / cgamma) - (chabb / cgamma) / (1 - chabb / cgamma) * cf[-1]
    kpf[0] = (1 - cikbar) * kpf[-1] + cikbar * invef[0] + cikbar * cgamma ^ 2 * csadjcost * qs[0]

    # Sticky-price and sticky-wage economy
    mc[0] = calfa * rk[0] + (1 - calfa) * w[0] - a[0]
    zcap[0] = rk[0] / cZcap
    rk[0] = w[0] + lab[0] - k[0]
    k[0] = kp[-1] + zcap[0]
    inve[0] = (inve[-1] + cbetabar * cgamma * inve[1] + pk[0] / (cgamma ^ 2 * csadjcost)) / (1 + cbetabar * cgamma) + qs[0]
    pk[0] = -r[0] + pinf[1] + b[0] / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) + crk / (crk + 1 - ctou) * rk[1] + (1 - ctou) / (crk + 1 - ctou) * pk[1]
    c[0] = (chabb / cgamma) / (1 + chabb / cgamma) * c[-1] + c[1] / (1 + chabb / cgamma) + (csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma)) * (lab[0] - lab[1]) - (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)) * (r[0] - pinf[1]) + b[0]
    y[0] = ccy * c[0] + ciy * inve[0] + g[0] + crkky * zcap[0]
    y[0] = cfc * (calfa * k[0] + (1 - calfa) * lab[0] + a[0])
    pinf[0] = (cbetabar * cgamma * pinf[1] + cindp * pinf[-1] + ((1 - cprobp) * (1 - cbetabar * cgamma * cprobp) / cprobp) / ((cfc - 1) * curvp + 1) * mc[0]) / (1 + cbetabar * cgamma * cindp) + spinf[0]
    w[0] = w[-1] / (1 + cbetabar * cgamma) + cbetabar * cgamma / (1 + cbetabar * cgamma) * w[1] + cindw / (1 + cbetabar * cgamma) * pinf[-1] - (1 + cbetabar * cgamma * cindw) / (1 + cbetabar * cgamma) * pinf[0] + cbetabar * cgamma / (1 + cbetabar * cgamma) * pinf[1] + (1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / ((1 + cbetabar * cgamma) * cprobw) / ((clandaw - 1) * curvw + 1) * (csigl * lab[0] + c[0] / (1 - chabb / cgamma) - (chabb / cgamma) / (1 - chabb / cgamma) * c[-1] - w[0]) + sw[0]
    r[0] = crpi * (1 - crr) * pinf[0] + cry * (1 - crr) * (y[0] - yf[0]) + crdy * (y[0] - yf[0] - y[-1] + yf[-1]) + crr * r[-1] + ms[0]

    # Exogenous processes with estimated innovation scales
    a[0] = crhoa * a[-1] + z_ea * ea[x]
    b[0] = crhob * b[-1] + z_eb * eb[x]
    g[0] = crhog * g[-1] + z_eg * eg[x] + cgy * z_ea * ea[x]
    qs[0] = crhoqs * qs[-1] + z_eqs * eqs[x]
    ms[0] = crhoms * ms[-1] + z_em * em[x]
    spinf[0] = crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]
    epinfma[0] = z_epinf * epinf[x]
    sw[0] = crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]
    ewma[0] = z_ew * ew[x]
    kp[0] = (1 - cikbar) * kp[-1] + cikbar * inve[0] + cikbar * cgamma ^ 2 * csadjcost * qs[0]

    # Measurement equations
    dy[0] = y[0] - y[-1] + ctrend
    dc[0] = c[0] - c[-1] + ctrend
    dinve[0] = inve[0] - inve[-1] + ctrend
    dwobs[0] = w[0] - w[-1] + ctrend
    pinfobs[0] = pinf[0] + constepinf
    robs[0] = r[0] + conster
    labobs[0] = lab[0] + constelab
end

@parameters Smets_Wouters_2007_linear begin
    ctou = 0.025
    clandaw = 1.5
    cg = 0.18
    curvp = 10.0
    curvw = 10.0

    cpie = 1 + constepinf / 100 | cpie
    cgamma = 1 + ctrend / 100 | cgamma
    cbeta = 1 / (1 + constebeta / 100) | cbeta
    clandap = cfc | clandap
    cbetabar = cbeta * cgamma ^ (-csigma) | cbetabar
    cr = cpie / (cbeta * cgamma ^ (-csigma)) | cr
    crk = cbeta ^ (-1) * cgamma ^ csigma - (1 - ctou) | crk
    cw = (calfa ^ calfa * (1 - calfa) ^ (1 - calfa) / (clandap * crk ^ calfa)) ^ (1 / (1 - calfa)) | cw
    cikbar = 1 - (1 - ctou) / cgamma | cikbar
    cik = cikbar * cgamma | cik
    clk = (1 - calfa) / calfa * crk / cw | clk
    cky = cfc * clk ^ (calfa - 1) | cky
    ciy = cik * cky | ciy
    ccy = 1 - cg - ciy | ccy
    crkky = crk * cky | crkky
    cwhlc = (1 / clandaw) * (1 - calfa) / calfa * crk * cky / ccy | cwhlc
    conster = (cr - 1) * 100 | conster
    cZcap = czcap / (1 - czcap) | cZcap

    calfa = 0.24
    csigma = 1.5
    cfc = 1.5
    cgy = 0.51
    csadjcost = 6.0144
    chabb = 0.6361
    cprobw = 0.8087
    csigl = 1.9423
    cprobp = 0.6
    cindw = 0.3243
    cindp = 0.47
    czcap = 0.2696
    crpi = 1.488
    crr = 0.8762
    cry = 0.0593
    crdy = 0.2347
    crhoa = 0.9977
    crhob = 0.5799
    crhog = 0.9957
    crhoqs = 0.7165
    crhoms = 0.0
    crhopinf = 0.0
    crhow = 0.0
    cmap = 0.0
    cmaw = 0.0
    constelab = 0.0
    constepinf = 0.7
    constebeta = 0.7420
    ctrend = 0.3982
    z_ea = 0.4618
    z_eb = 0.1818513
    z_eg = 0.6090
    z_eqs = 0.46017
    z_em = 0.2397
    z_epinf = 0.1455
    z_ew = 0.2089
end
