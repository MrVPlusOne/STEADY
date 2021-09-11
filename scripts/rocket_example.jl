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
times = collect(0.0:0.1:10.0)
params = (
    drag = 0.1,
    mass = 1.5,
    rot_mass = 1.0,
    rot_drag = 0.2,
    length = 0.5,
    gravity = 1.25,
)
n_landmarks = 3
landmarks = [
    @SVector[-12., -4],
    @SVector[-2., -7],
    @SVector[4., -4],
]
others = (landmarks = landmarks,)
target_pos = @SVector[-3.0, -2.]
x₀ = (pos=@SVector[0.0, 0.0], θ=45°,)
x₀′ = (pos′=@SVector[0.25, 0.1], θ′=-5°,)
ex_data = Rocket2D.generate_data(x₀, x₀′, params, others, times; noise_scale, target_pos)

Rocket2D.plot_data(ex_data, landmarks, "Truth")
## program enumeration
shape_env = ℝenv()
comp_env = ComponentEnv()
can_grow = false
components_scalar_arithmatic!(comp_env; can_grow)
components_transcendentals!(comp_env; can_grow)
components_vec2!(comp_env; can_grow)

vdata = Rocket2D.variable_data(n_landmarks)
prog_logp(comps) = log(0.5) * sum(ast_size.(comps)) 

pruner = NoPruner()
senum = synthesis_enumeration(
    vdata, Rocket2D.action_vars(), comp_env, 5, pruner,
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
        evals_per_program=10,
        optim_options = Optim.Options(x_abstol=1e-3),
        n_threads=1,
    )
end
display(syn_result)
show_top_results(syn_result, 5)
##