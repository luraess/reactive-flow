using Plots, ElasticArrays, Printf

@views avx(A) = (A[1:end-1] .+ A[2:end]) .* 0.5

@views function main()
    # real values
    k_r     = 1e-12
    ηf_r    = 1e4
    ϕ_bg    = 1e-2
    η_r     = 1e18
    ρs      = 2700
    ρf      = 2300
    g       = 9.81
    # independent physics
    @show lc      = sqrt(k_r / ηf_r * η_r / ϕ_bg) # 1.0  # m
    @show Δρg     = (ρs - ρf) * g # 1.0  # Pa/m
    @show η_ϕbg   = η_r / ϕ_bg # 1.0  # Pa*s
    # scales
    psc     = Δρg * lc
    tsc     = η_ϕbg / psc
    # non-dimensional numbers
    lx_lc   = 100.0
    w_lc    = 5.0
    # ϕ_bg    = 0.01
    ϕA      = 0.1
    npow    = 3
    # dependent physics
    lx      = lx_lc * lc
    w       = w_lc * lc
    k_ηf0   = lc^2 / η_ϕbg
    dt      = 1e-3 * tsc
    # numerics
    nx      = 1000
    nt      = 50
    maxiter = 50nx
    ϵtol    = [1e-8, 1e-8]
    cfl     = 1 / 1.1
    ncheck  = ceil(Int, 0.25nx)
    # preprocessing
    dx      = lx / nx
    xc      = LinRange(-lx / 2 - dx / 2, lx / 2 + dx / 2, nx)
    vsdτ    = cfl * dx
    # init
    ϕ       = ϕ_bg .+ ϕA .* exp.(.-((xc .+ lx / 3) ./ w) .^ 2)
    ϕ_i     = copy(ϕ)
    Pe      = zeros(nx)
    qD      = zeros(nx - 1)
    η_ϕ     = zeros(nx)
    k_ηf    = zeros(nx)
    RPe     = zeros(nx - 2)
    RqD     = zeros(nx - 1)
    η_ϕτ    = zeros(nx)
    βf_dτ   = zeros(nx)
    k_ηfτ   = zeros(nx - 1)
    ρ_dτ    = zeros(nx)
    re      = zeros(nx)
    lc_loc  = zeros(nx)
    # action
    for it = 1:nt
        @printf("it = %d\n", it)
        iter = 1
        errs = 2.0 .* ϵtol
        errs_evo = ElasticArray{Float64}(undef, 2, 0)
        iters_evo = Float64[]
        while any(errs .> ϵtol) && iter <= maxiter
            # material properties
            η_ϕ    .= η_ϕbg .* (ϕ_bg ./ ϕ)
            k_ηf   .= k_ηf0 .* (ϕ ./ ϕ_bg) .^ npow
            # numerical parameter update
            lc_loc .= sqrt.(k_ηf .* η_ϕ)
            re     .= π .+ sqrt.(π .^ 2 .+ (lx ./ lc_loc) .^ 2)
            ρ_dτ   .= re .* η_ϕ ./ lx ./ dx ./ cfl
            βf_dτ  .= 1 / vsdτ^2 ./ ρ_dτ
            η_ϕτ   .= 1.0 ./ (βf_dτ .+ 1.0 ./ η_ϕ)
            k_ηfτ  .= 1.0 ./ (avx(ρ_dτ) .+ 1.0 ./ avx(k_ηf))
            # update of physical fields
            RPe    .= .-Pe[2:end-1] ./ η_ϕ[2:end-1] .+ diff(qD) ./ dx
            Pe[2:end-1] .+= RPe .* η_ϕτ[2:end-1]
            RqD    .= .-qD ./ avx(k_ηf) .+ diff(Pe) ./ dx .+ Δρg
            qD    .+= RqD .* k_ηfτ
            if iter % ncheck == 0
                errs[1] = maximum(abs.(RPe))
                errs[2] = maximum(abs.(RqD))
                append!(errs_evo, errs)
                push!(iters_evo, iter / nx)
                @printf("  iter = %d, iter/nx = %1.3e, errs = [ %1.3e %1.3e ]\n", iter, iter / nx, errs...)
            end
            iter += 1
        end
        ϕ .-= dt * Pe ./ η_ϕ
        p1 = plot(xc, [ϕ_i, ϕ]; title="Porosity", xlabel="x-direction", label=["Init" "φ"])
        p2 = plot(xc, Pe; title="Effective Pressure", xlabel="x-direction", label="Pe")
        p3 = plot(iters_evo, errs_evo'; title="Residual evolution", yaxis=:log10, marker=:circle, xlabel="Iter/nx", label=["RPe" "RqD"])
        display(plot(p1, p2, p3; layout=(3, 1)))
    end
end

main()

# doc
# Pe = Pe + RPe./βf_dτ # explicit update
