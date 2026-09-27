@model Gali_3eq_linear begin
    x[0] = x[1] - (1 / sigma) * (i[0] - pi[1] - rn[0])

    pi[0] = beta * pi[1] + kappa * x[0] + u[0]

    i[0] = rho_i * i[-1] + (1 - rho_i) * (phi_pi * pi[0] + phi_x * x[0]) + std_i * eps_i[x]

    rn[0] = rho_rn * rn[-1] + std_rn * eps_rn[x]

    u[0] = rho_u * u[-1] + std_u * eps_u[x]

    y_obs[0] = x[0]

    pi_obs[0] = pi[0]

    i_obs[0] = i[0]
end


@parameters Gali_3eq_linear begin
    sigma = 1.0

    beta = 0.99

    kappa = 0.10

    phi_pi = 1.5

    phi_x = 0.125

    rho_i = 0.7

    rho_rn = 0.6

    rho_u = 0.5

    std_i = 0.0025

    std_rn = 0.01

    std_u = 0.005
end
