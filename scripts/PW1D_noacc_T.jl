using Plots, ElasticArrays, Printf

@views avx(A) = (A[1:end-1] .+ A[2:end]) .* 0.5

# switch_advect = WENA_5()!!!

@views function main()
    # independent physics
    lc      = 1.0 #sqrt(k_r / ηf_r * η_r / ϕ_bg) # m
    Δρg     = 1.0 #(ρs - ρf) * g # Pa/m
    η_ϕbg   = 1.0 #η_r / ϕ_bg # Pa*s
    # scales
    psc     = Δρg * lc
    tsc     = η_ϕbg / psc
    # non-dimensional numbers
    lx_lc   = 100.0
    w_lc    = 5.0
    ϕ_bg    = 0.01
    ϕA      = 0.1
    npow    = 3
    T_bg    = 0.0
    TA      = 1.0
    # dependent physics
    lx      = lx_lc * lc
    w       = w_lc * lc
    k_ηf0   = lc^2 / η_ϕbg
    λ       = 1.0
    dt      = 1e-3 * tsc / 20
    # numerics
    nx      = 200
    nt      = 1000
    maxiter = 100nx
    ϵtol    = [1e-5 1.0]
    ncheck  = ceil(Int, 40nx)
    # preprocessing
    dx      = lx / nx
    xc      = LinRange(-lx / 2 - dx / 2, lx / 2 + dx / 2, nx)
    # init
    ϕ       = ϕ_bg .+ ϕA .* exp.(.-((xc .+ lx / 3) ./ w) .^ 2)
    T       = T_bg .+ TA .* exp.(.-((xc .+ lx / 3) ./ w) .^ 2)
    # T = T_bg .* ones(nx)
    # T[-lx / 2.8 .< xc .< -lx / 3.2] .= T_bg .+ TA
    ϕ_i     = copy(ϕ)
    ϕ_old   = copy(ϕ)
    T_old   = copy(T)
    Pe      = zeros(nx)
    qD      = zeros(nx - 1)
    η_ϕ     = zeros(nx)
    k_ηf    = zeros(nx)
    dϕdt    = zeros(nx)
    dTdt    = zeros(nx)
    RPe     = zeros(nx - 2)
    dτ_β    = zeros(nx - 2)
    qTx     = zeros(nx - 1)
    # ρCp_f * ϕ * T + ρCp_s * (1-ϕ) * T -> ρCp_t * T
    sum_T_i = sum(T)
    # action
    for it = 1:nt
        @printf("it = %d\n", it)
        ϕ_old .= ϕ
        T_old .= T
        iter = 1
        errs = 2.0 .* ϵtol
        errs_evo = ElasticArray{Float64}(undef, 2, 0)
        iters_evo = Float64[]
        while any(errs .> ϵtol) && iter <= maxiter
            # material properties
            η_ϕ  .= η_ϕbg .* (ϕ_bg ./ ϕ)
            k_ηf .= k_ηf0 .* (ϕ ./ ϕ_bg) .^ npow
            # update of physical fields
            qD   .= avx(k_ηf) .* (diff(Pe) ./ dx .+ Δρg)
            RPe  .= .-Pe[2:end-1] ./ η_ϕ[2:end-1] .+ diff(qD) ./ dx
            dτ_β .= dx^2 ./ k_ηf[2:end-1] ./ 3.1
            Pe[2:end-1] .+= RPe .* dτ_β
            @. dϕdt = -Pe / η_ϕ
            @. ϕ = ϕ_old + dt * dϕdt

            # d(ρcp_f * ϕ * T)/dt + d(ρcp_s * (1-ϕ) * T)/dt + ∇(ρcp_f * ϕ * T * Vf) + ∇(ρcp_s * (1-ϕ) * T * Vs) - ∇(λtot * ∇T) = 0

            # T * d(ρcp_f * ϕ)/dt     + (ρcp_f * ϕ) * d(T)/dt +
            # T * d(ρcp_s * (1-ϕ))/dt + (ρcp_s * (1-ϕ)) * d(T)/dt +
            # T * ∇(ρcp_f * ϕ * Vf)     + ρcp_f * ϕ * Vf * ∇(T) +
            # T * ∇(ρcp_s * (1-ϕ) * Vs) + ρcp_s * (1-ϕ) * Vs * ∇(T) - ∇(λtot * ∇T) = 0

            # T * (d(ρcp_f * ϕ)/dt + d(ρcp_s * (1-ϕ))/dt + ∇(ρcp_f * ϕ * Vf) + ∇(ρcp_s * (1-ϕ) * Vs)) + # -> 0
            # d(T)/dt * ((ρcp_f * ϕ) + (ρcp_s * (1-ϕ))) + # -> d(T)/dt * ρcp_t
            # ρcp_f * qD * ∇(T) + ρcp_t * Vs * ∇(T) - ∇(λtot * ∇T) = 0

            # D(T)/Dt * ρcp_t + # -> D(T)/Dt -> dT/dt + Vs * ∇(T)
            # ρcp_f * qD * ∇(T) - ∇(λtot * ∇T) = 0

            # D(T)/Dt + ρcp_f /  ρcp_t * qD * ∇(T) - λtot /  ρcp_t * ∇(∇T) = 0

            # energy / temperature
            ret = 1e1
            qTx .= .-λ .* diff(T) ./ dx
            dTdt[2:end-1] .= .-diff(qTx) ./ dx
            dTdt[1:end-1] .-= ret .* min.(qD, 0.0) .* diff(T) ./ dx
            dTdt[2:end]   .-= ret .* max.(qD, 0.0) .* diff(T) ./ dx
            T .= T_old .+ dt .* dTdt
            if iter % ncheck == 0
                errs[1] = maximum(abs.(RPe))
                errs[2] = (sum(T) .- sum_T_i) ./ sum_T_i
                append!(errs_evo, errs)
                push!(iters_evo, iter / nx)
                @printf("  iter = %d, iter/nx = %1.3e, err = [ %1.3e %1.3e ] \n", iter, iter / nx, errs...)
            end
            iter += 1
        end
        if it % 10 == 0
        # visualisation
        p1 = plot([ϕ_i, ϕ], xc; title="Porosity", ylabel="depth", label=["Init" "φ"])
        p2 = plot(Pe, xc; title="Effective Pressure", label="Pe")
        p3 = plot(T, xc; xlabel="$(round(errs[2], sigdigits=4))", title="Temperature", label="T")
        display(plot(p1, p2, p3; layout=(1, 3)))
        end
    end
end

main()
