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
times = collect(0.0:0.1:20.0)
params = (
    drag = 0.1,
    mass = 1.5,
    rot_mass = 1.0,
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
target_pos = @SVector[-2.5, -2.5]
x₀ = (pos=@SVector[0.0, 0.0], θ=45°,)
x₀′ = (pos′=@SVector[0.25, 0.1], θ′=-5°,)
ex_data = Rocket2D.generate_data(x₀, x₀′, params, others, times; noise_scale, target_pos)
Rocket2D.plot_data(ex_data, "Truth")
## program enumeration
shape_env = ℝenv()
comp_env = ComponentEnv()
can_grow = true
components_scalar_arithmatic!(comp_env; can_grow)
components_transcendentals!(comp_env; can_grow)
components_vec2!(comp_env; can_grow)

vdata = Rocket2D.variable_data(n_landmarks)
prog_logp(comps) = log(0.5) * sum(ast_size.(comps)) 

# pruner = NoPruner()
pruner = RebootPruner(; comp_env.rules)
senum = synthesis_enumeration(
    vdata, Rocket2D.action_vars(), comp_env, 5; pruner, type_pruning=true,
)
display(senum)
## perform MAP synthesis
syn_result = @time let
    observations = ex_data.observations
    map_synthesis(
        senum,
        shape_env,
        ex_data.actions, ex_data.times, 
        prog_logp, 
        (states, params) -> Rocket2D.data_likelihood(states, params, observations; noise_scale), 
        evals_per_program=20,
        optim_options = Optim.Options(x_abstol=1e-3),
        n_threads=6,
    )
end
display(syn_result)
show_top_results(syn_result, 5)
##
map_data = syn_result.sorted_results[1].MAP_est
Rocket2D.plot_data(map_data, "MAP Estimate") |> display

syn_result.errored_programs
##

senum = synthesis_enumeration(
    vdata, Rocket2D.action_vars(), comp_env, 4; pruner,
)
types_needed, _ = synthesis_enumeration_staged(
    vdata, Rocket2D.action_vars(), comp_env, 4,
)

