# this script needs to be run inside the module SEDL
## imports
using DataFrames
using StatsPlots
using BenchmarkTools
## test enumeration
shape_env = ℝenv()
env = ComponentEnv()
components_scalar_arithmatic!(env)
components_vec2!(env)
components_transcendentals!(env)

l = Var(:l, ℝ, PUnits.Length)
x = Var(:x, ℝ2, PUnits.Length)
v = Var(:v, ℝ2, PUnits.Speed)
θ = Var(:θ, ℝ, PUnits.Angle)
steer = Var(:steer, ℝ, PUnits.AngularSpeed)
wall_x = Var(:wall_x, ℝ, PUnits.Length)

result = bottom_up_enum(env, [x, l, θ], 6)
## test pruning
display(env.rules)
before = collect(result[l.type])
after, pruned, report = let 
    n = length(before)
    params = SaturationParams(
        scheduler=Metatheory.Schedulers.SimpleScheduler,
        # scheduler=Metatheory.Schedulers.ScoredScheduler,
        # schedulerparams=(n, 5),
        timeout=30, eclasslimit=10n, enodelimit=30n, matchlimit=100n)
    pstate = PruningState{TAST}()
    prune_redundant!(pstate, env.rules, before, params)
end
report
##
length(before)
DataFrame(pruned)
after
## test compilation
prog = Iterators.drop(result[ℝ2], 10) |> first
f_comp = compile(prog, shape_env, env)
f_comp(((x=@SVector[1.0, 2.0], l=1.4, θ=2.1)))
## test compilation speed
let
    result = bottom_up_enum(env, [x, l, θ], 5)
    programs = collect(Iterators.take(result[ℝ], 500))
    stats = map(programs) do p
        size = ast_size(p)
        compile_time = @elapsed let
            f_comp = compile(p, shape_env, env)
            f_comp((x=@SVector[1.0, 2.0], l=1.4, θ=2.1))
        end
        (;size, compile_time)
    end |> DataFrame
    time_vs_size = combine(groupby(stats, :size), :compile_time => mean => :compile_time)
    @df time_vs_size plot(:size, :compile_time)
end
##
# test simulation
@time let
    s = (x=1.0,)
    s′ = (x′=0.0,)
    params = (a = 1.0,)
    times = range(0, 10, length=100)
    actions = map(_ -> (f=2.0,), times)
    f_s′′= ((args) -> (-1 * args.a * args.x),)
    simulate(s, s′, f_s′′, params, times, actions)
end

gen_traj(N) = begin
    s = (x=1.0,)
    s′ = (x′=0.0,)
    params = (a = 1.0,)
    times = range(0, 10, length=N) |> collect
    actions = map(_ -> (f=2.0,), times)
    acc = CompiledFunc(x, :(1+1), (args) -> (-args.a * args.x - 0.1 * args.x′))
    f_s′′= (acc,)
    simulate(s, s′, f_s′′, params, times, actions) |> specific_elems
end

gen_traj(10)
# @benchmark gen_traj(100)

begin
    plot(gen_traj(100), label="N=100")
    plot!(gen_traj(10_000), label="N=10_000")
end
##
# test map_estimate
compute_solution(; only_prior::Bool, N=200) = begin
    s = (x=Normal(1.0),)
    s′ = (x′=Normal(0.0),)
    params = (drag = Uniform(0.0, 0.5),)
    times = range(0, 10, length=N)
    actions = map(_ -> (f=2.0,), times)
    f_s′′= ((args) -> (-args.x - args.drag * args.x′),)
    function likelihood(traj, p)
        if only_prior
            0.0
        else
            -10((traj[1].x - 1)^2 + (5traj[end].x)^2 + (5traj[end].x′)^2)
        end
    end
    map_trajectory(s, s′, f_s′′, params, times, actions, likelihood, Optim.Options())
end

let 
    result_priror = compute_solution(only_prior=true)
    println("Prior params: $(result_priror.params)")
    plot(specific_elems(result_priror.traj), label="Prior")
    result_post = compute_solution(only_prior=false)
    println("Posterior params: $(result_post.params)")
    plot!(specific_elems(result_post.traj), label="Posterior")
end
## example synthesis problem
noise_scale = 1.0
times = collect(range(0.0, 5.0, length=50))
params = (drag=0.1, mass=1.5)
others = (wall=7.0,)
x₀ = (pos=0.0,)
x₀′ = (pos′=0.5,)
ex_data = Car1D.generate_data(x₀, x₀′, (Car1D.acceleration_f,), params, others, times; noise_scale)
Car1D.plot_data(ex_data, "Truth")
## run synthesis
shape_env = ℝenv()
env = ComponentEnv()
components_scalar_arithmatic!(env)
components_transcendentals!(env)

vdata = Car1D.variable_data()
prog_logp(comps) = log(0.5) * sum(ast_size.(comps))  # weakly penealize larger programs

syn_result = @time let 
    observations = ex_data.observations
    map_synthesis(
        shape_env, env, vdata, Car1D.action_vars(),
        ex_data.actions, ex_data.times, 
        prog_logp, 
        (states, params) -> Car1D.data_likelihood(states, params, observations; noise_scale), 
        max_size=6,
        evals_per_program=10,
        optim_options = Optim.Options(x_abstol=1e-3),
        n_threads=6,
    )
end
##
show_top_results(syn_result, 5)
let 
    (; observations, actions, times) = ex_data
    post_data = merge(syn_result.best_result.MAP_est, (;observations, actions, times))
    Car1D.plot_data(post_data, "MAP")
end
##