using OrdinaryDiffEq, ModelingToolkit
using Plots


function example()
    local t, m, γ
    local x, v, u

    @parameters t m γ
    @variables x(t) v(t) u(t)
    D = Differential(t)

    eqs = [
        D(x) ~ v,
        D(v) ~ (u - γ * v) / m,
        D(u) ~ 0,
    ]

    function ctrl_callback(int)
        int.u[v]
    end

    sys = ODESystem(eqs) |> ode_order_lowering

    s0 = [x => 0.0, v => 1.0, u => 0.0]
    p = [m => 2.5, γ => 0.2]
    tspan = (0.0,100.0)
    prob = ODEProblem(sys,s0,tspan,p,jac=true)
    # solve(prob,Tsit5())
    sys, init(prob, Tsit5())
end
# plot(sol,vars=(x, v))
@nonamespace sys.v
sys, sol = example()
ModelingToolkit.varmap_to_vars
propertynames(sol)
(@nonamespace sys.x) in parameters(sys)
sol.u[@nonamespace sys.x] = 5
propertynames(sys)
parameters(sys)
states(sys)
sol[@nonamespace sys.v]
sol.u[x]