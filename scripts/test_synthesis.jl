# this script needs to be run inside the module SEDL
## imports
using DataFrames
using StatsPlots
using BenchmarkTools
## test enumeration and pruning
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

max_prog_size = 7
pruners = [
    NoPruner(), 
    RebootPruner(; env.rules, only_postprocess=false), 
    # IncrementalPruner(; env.rules),
]
results_timed = map(pruners) do pruner
    @timed enumerate_terms(env, [x, l, θ], max_prog_size; pruner)
end
results = (v -> v.value).(results_timed)
for (p, (;value, time)) in zip(pruners, results_timed)
    println("** $(typeof(p)) took $(time)s.")
    println("n_programs with type ℝ: ", count(_ -> true, value[ℝ]))
    display(value)
end
## test compilation
prog = Iterators.drop(results[1][ℝ2], 10) |> first
f_comp = compile(prog, shape_env, env)
f_comp(((x=@SVector[1.0, 2.0], l=1.4, θ=2.1)))
## test compilation speed
let
    result = enumerate_terms(env, [x, l, θ], 5)
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
## test simulation
gen_traj(N) = begin
    s = (x=1.0,)
    s′ = (x′=0.0,)
    params = (a = 1.0,)
    times = range(0, 10, length=N) |> collect
    actions = map(_ -> (f=2.0,), times)
    f_s′′= ((args) -> (x′′ = -1 * args.a * args.x, ))
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
    s = (x=SNormal(1.0),)
    s′ = (x′=SNormal(0.0),)
    params = (drag = SUniform(0.0, 0.5),)
    times = range(0, 10, length=N)
    actions = map(_ -> (f=2.0,), times)
    f_s′′= (args) -> (x′′= -args.x - args.drag * args.x′,)
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
params = (mass=1.5, drag=0.1, )
others = (wall=7.0,)
x₀ = (pos=0.0,)
x′₀ = (pos′=0.5,)
ex_data = Car1D.generate_data(x₀, x′₀, params, others, times; noise_scale)
Car1D.plot_data(ex_data, "Truth")
## perform enumeration for synthesis
shape_env = ℝenv()
comp_env = ComponentEnv()
components_scalar_arithmatic!(comp_env, can_grow=false)
# components_transcendentals!(comp_env)

vdata = Car1D.variable_data()
prior_p = logpdf(to_distribution(vdata), (;x₀, x′₀, params, others))
if !isfinite(prior_p)
    error("prior_p = $prior_p, some value may be out of its support.")
end
sketch = no_sketch(vdata.state′′_vars)
prog_logp(comps) = log(0.5) * sum(ast_size.(comps))  # weakly penealize larger programs

pruner = RebootPruner(rules=comp_env.rules)
# pruner = NoPruner()
senum = synthesis_enumeration(
    vdata, sketch, Car1D.action_vars(), comp_env, 5; pruner)
let rows = map(senum.enum_result.pruned) do r
        (; r.pruned, r.by)#, explain=join(r.explain, " ; "))
    end
    show(DataFrame(rows), truncate=100)
    println()
end
display(senum)
## perform MAP sythesis
syn_result = let 
    observations = ex_data.observations
    map_synthesis(
        senum,
        shape_env,
        ex_data.actions, ex_data.times, 
        prog_logp, 
        (states, params) -> Car1D.data_likelihood(states, params, observations; noise_scale), 
        evals_per_program=10,
        optim_options = Optim.Options(x_abstol=1e-3),
        n_threads=4,
    )
end
display(syn_result)
show_top_results(syn_result, 5)
##
let 
    (; observations, actions, times) = ex_data
    post_data = merge(syn_result.best_result.MAP_est, (;observations, actions, times))
    Car1D.plot_data(post_data, "MAP")
end
## test prunning correctness
example_pruning_check()
##