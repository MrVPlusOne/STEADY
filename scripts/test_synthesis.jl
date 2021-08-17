# this script needs to be run inside the module SEDL
##
env = ComponentEnv()
components_scalar_arithmatic!(env)
components_vec2!(env)
components_transcendentals!(env)
env

l = Var(:l, ℝ, :L => 1)
x = Var(:x, ℝ2, :L => 1)
v = Var(:v, ℝ2, :L => 1, :T => -1)
θ = Var(:θ, ℝ)
steer = Var(:steer, ℝ, :T => -1)
wall_x = Var(:wall_x, ℝ, :L => 1)

derivative(x)

result = bottom_up_enum(env, [x, l, θ], 7)
##
# test compilation
prog = Iterators.drop(result[ℝ2], 10) |> first
f_comp = compile(prog, [x, l, θ], ℝenv(), env)
f_comp($((x=@SVector[1.0, 2.0], l=1.4, θ=2.1)))
##
# test compilation speed
using DataFrames
using StatsPlots
using BenchmarkTools
programs = collect(Iterators.take(result[ℝ], 500))
stats = map(programs) do p
    size = ast_size(p)
    compile_time = @elapsed let
        f_comp = compile(p, [x, l, θ], ℝenv(), env)
        f_comp((x=@SVector[1.0, 2.0], l=1.4, θ=2.1))
    end
    (;size, compile_time)
end |> DataFrame
time_vs_size = combine(groupby(stats, :size), :compile_time => mean => :compile_time)
@df time_vs_size plot(:size, :compile_time)
##
# test simulation
@time let
    s = (x=1.0,)
    s′ = (x′=0.0,)
    params = (a = 1.0,)
    times = range(0, 10, length=100)
    actions = map(_ -> (f=2.0,), times)
    f_s′′= (x′′ = (args) -> (-1 * args.a * args.x),)
    Main.@code_warntype simulate(s, s′, f_s′′, params, times, actions)
end

gen_traj(N) = begin
    s = (x=1.0,)
    s′ = (x′=0.0,)
    params = (a = 1.0,)
    times = range(0, 10, length=N)
    actions = map(_ -> (f=2.0,), times)
    f_s′′= (x′′ = (args) -> (-args.a * args.x - 0.1 * args.x′),)
    simulate(s, s′, f_s′′, params, times, actions)
end

begin
    plot(gen_traj(100), label="N=100")
    plot!(gen_traj(10_000), label="N=10_000")
end
##
# test synthesis
n_steps = 10
vdata = VariableData(
    states = Dict([
        x => (MvNormal(zeros(2), ones(2)*0.1), MvNormal(zeros(2), ones(2))),
        θ => (Normal(pi/2, 0.1), Normal(0.0, 0.5)),
    ]),
    actions = Dict([
        steer => fill(1.0, n_steps),
    ]),
    dynamics_params = Dict([
        l => Uniform(0.0, 2.0),
    ]),
    others = Dict([
        wall_x => Uniform(0.0, 50.0)
    ]),
)
map_synthesis(env, vdata, identity, identity, 4)
##
