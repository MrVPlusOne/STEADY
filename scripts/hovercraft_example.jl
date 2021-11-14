##-----------------------------------------------------------
using Distributions
using StatsPlots
using DrWatson
import Random

StatsPlots.default(dpi=300, legend=:outerbottom)

# generate data
Random.seed!(123)
landmarks = @SVector[
    @SVector[-1.0, 2.5], @SVector[1.0, -1.0],
    @SVector[8.0, -5.5], @SVector[14.0, 6.0], @SVector[16.0, -7.5]]
scenario = HovercraftScenario(LandmarkInfo(; landmarks))
times = collect(0.0:0.1:14)
obs_frames = 1:5:length(times)
true_params = (; 
    mass = 1.5, drag_x = 0.08, drag_y=0.12, rot_mass=1.5, rot_drag=0.07, sep=0.81,
    σ_v=0.04, σ_ω=0.03)
params_guess = (; 
    mass = 1.65, drag_x = 0.04, drag_y=0.1, rot_mass=1.66, rot_drag=0.09, sep=0.87,
    σ_v=0.1, σ_ω=0.1)

function manual_control()
    pert(x) = x + 0.1randn()
    @unzip times, ul_seq, ur_seq = [
        (t=0.0, ul=0.0, ur=0.0),
        (t=pert(0.5), ul=pert(1.0), ur=pert(0.4)),
        (t=pert(2.0), ul=0.0, ur=0.0),
        (t=pert(3.0), ul=pert(0.5), ur=pert(0.5)),
        (t=pert(5.0), ul=pert(1.1), ur=pert(0.5)),        
        (t=pert(6.0), ul=0.0, ur=0.0),        
        (t=pert(9.0), ul=pert(0.5), ur=pert(1.0)),   
        (t=pert(11.0), ul=0.0, ur=pert(0.4)),     
        (t=12.0, ul=0.0, ur=0.0),     
        (t=15.0, ul=0.0, ur=0.0),
    ]
    ul_f = LinearInterpolation(times, ul_seq)
    ur_f = LinearInterpolation(times, ur_seq)
    if rand() < 0.6
        ul_f, ur_f = ur_f, ul_f
    end
    (s, obs, t::Float64) -> begin
        (ul = ul_f(t), ur = ur_f(t))
    end
end

n_runs = 6
n_fit_trajs = 15
setups = map(1:n_runs) do i 
    x0 = (
        pos=@SVector[0.5+randn(), 0.5+randn()], 
        vel=@SVector[0.25+0.3randn(), 0.0+0.2randn()], 
        θ=randn()°, 
        ω=0.2randn(),
    )
    ScenarioSetup(times, obs_frames, x0, manual_control())
end

comp_env = ComponentEnv()
components_scalar_arithmatic!(comp_env, can_grow=true)
# components_special_functions!(comp_env, can_grow=true)

# comps_guess = let 
#     vars = variables(scenario)
#     (; ul, mass, rot_mass, sep, loc_vx, drag_x, loc_vy, drag_y, ω, rot_drag) = vars
#     f_zero = get_component(comp_env, :zero)
#     # see if it stays with this correct dynamics
#     # (f_x=-loc_vx * drag_x, f_y=-loc_vy * drag_y, f_θ = -ω * rot_drag,)
#     (f_x=f_zero(ul), f_y=f_zero(ul), f_θ = f_zero(ul) * sep,)
# end
##-----------------------------------------------------------
# simulate the scenario
save_dir=datadir("sims/hovercraft")
true_motion_model = let 
    sketch=dynamics_sketch(scenario) 
    core=dynamics_core(scenario)
    to_p_motion_model(core, sketch)(true_params)
end
sim_result = simulate_scenario(scenario, true_motion_model, setups; save_dir);
##-----------------------------------------------------------
# test fitting the trajectories
let
    shape_env = ℝenv()
    sketch = sindy_sketch(scenario)
    basis = [compile(e, shape_env, comp_env) for e in sketch.input_vars]
    algorithm = SindySynthesis(basis, sketch, STLSOptimizer(0.01))
    particle_sampler = ParticleFilterSampler(
        n_particles=60_000,
        n_trajs=100,
    )
    test_scenario(scenario, sim_result, algorithm, particle_sampler)
end
##-----------------------------------------------------------
# plot and save the results
summary_dir=joinpath(save_dir, "summary") |> mkdir
open(joinpath(summary_dir, "iter_result.txt"), "w") do io
    println(io, iter_result)
end
let param_plt = plot_params(iter_result)
    display(param_plt)
    savefig(param_plt, joinpath(summary_dir, "params.svg"))
end
let perf_plot = plot(iter_result, start_idx=10)
    display(perf_plot)
    savefig(perf_plot, joinpath(summary_dir, "performance.svg"))
end
##-----------------------------------------------------------
# analyze the drag strength
function core_stats(input)
    (; loc_vx, loc_vy, ω, ul, ur, mass, drag_x, drag_y, rot_mass, rot_drag) = input
    f_x = drag_x * loc_vx
    f_y = drag_y * loc_vy
    f_θ = ω * rot_drag
    (; ul, ur, f_x, f_y, f_θ)
end

core_outputs = transform_sketch_inputs(core_stats, dynamics_sketch(scenario), 
    scenario_result.ex_data_list[1], true_params)

@df DataFrame(core_outputs) plot(times, [:ul :ur :f_x :f_y :f_θ])
##-----------------------------------------------------------
