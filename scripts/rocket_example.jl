# this script needs to be run inside the module SEDL
## imports
using DataFrames
using StatsPlots
using BenchmarkTools
import Random
using StaticArrays
## generate data
Random.seed!(123)

noise_scale = 1.0
times = collect(0.0:0.1:10)
params = (
    mass = 1.5,
    rot_mass = 1.0,
    drag = 0.1,
    rot_drag = 0.2,
    length = 0.5,
    gravity = @SVector[0., -1.25],
)
n_landmarks = 3
landmarks = [
    @SVector[-12., -4],
    @SVector[-2., -7],
    @SVector[4., -4],
]
others = (landmarks = landmarks,)
target_pos = @SVector[0.0, 0.0]
x₀ = (pos=@SVector[-5, -4.0], θ=45°,)
x′₀ = (pos′=@SVector[0.25, 0.1], θ′=-5°,)
ex_data = Rocket2D.generate_data(x₀, x′₀, params, others, times; noise_scale, target_pos)
Rocket2D.plot_data(ex_data, "Truth") |> display
## program enumeration
shape_env = ℝenv()
comp_env = ComponentEnv()
can_grow = true
components_void!(comp_env)
components_scalar_arithmatic!(comp_env; can_grow)
components_special_functions!(comp_env; can_grow)
components_vec2!(comp_env; can_grow)

vdata = Rocket2D.variable_data(n_landmarks, x₀)
prior_p = logpdf(to_distribution(vdata), (;x₀, x′₀, params, others))
if !isfinite(prior_p)
    error("prior_p = $prior_p, some value may be out of its support.")
end

# sketch = no_sketch(vdata.state′′_vars)
# sketch = Rocket2D.ground_truth_sketch()
sketch = Rocket2D.simple_sketch()

# pruner = NoPruner()
pruner = RebootPruner(; comp_env.rules)
senum = synthesis_enumeration(
    vdata, sketch, Rocket2D.action_vars(), comp_env, 5; pruner, type_pruning=true,
)
display(senum)
## perform MAP synthesis
data_thining = 5
syn_result = @time let 
    map_synthesis(
        senum,
        shape_env,
        ex_data.actions, ex_data.times, 
        prog_size_prior(0.5), 
        Rocket2D.data_likelihood(ex_data.observations; noise_scale, data_thining), 
        evals_per_program=10,
        trials_per_eval=5,
        optim_options=Optim.Options(
            f_abstol=1e-3, outer_f_abstol=1e-3, iterations=2000, outer_iterations=4),
        use_bijectors=true,
        use_distributed=true,
        n_threads=1,
    )
end
display(syn_result)
show_top_results(syn_result, 5)
map_data = syn_result.sorted_results[1].MAP_est
let 
    p = Rocket2D.plot_data(ex_data, "Truth")
    Rocket2D.plot_data!(p, map_data, "Estimate") |> display
end
syn_result.sorted_results[1].MAP_est.others |> dump
syn_result.errored_programs
##