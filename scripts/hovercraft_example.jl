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
obs_frames = 1:4:length(times)
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
n_fit_trajs = 10
setups = map(1:n_runs) do i 
    x0 = (
        pos=@SVector[0.5+randn(), 0.5+randn()], 
        vel=@SVector[0.25+0.3randn(), 0.0+0.2randn()], 
        θ=randn()°, 
        ω=0.2randn(),
    )
    ScenarioSetup(times, obs_frames, x0, manual_control())
end
nothing
##-----------------------------------------------------------
# simulate the scenario
save_dir=datadir("sims/hovercraft")
old_motion_model = let 
    sketch=dynamics_sketch(scenario) 
    core=dynamics_core(scenario)
    @show core
    to_p_motion_model(core, sketch)(true_params)
end
println("Ground truth motion model:")
true_comps = OrderedDict(pairs(sindy_core(scenario, true_params)))
display(true_comps)

new_motion_model = let
    sketch = sindy_sketch(scenario)
    core = sindy_core(scenario, true_params)
    println("Compiled sindy model:") 
    for f in core
        display(f.μ_f.julia)
    end
    sindy_motion_model(sketch, core)
end
sim_result = simulate_scenario(scenario, new_motion_model, setups; save_dir)
nothing
##-----------------------------------------------------------
# test fitting the trajectories
algorithm = let
    sketch = sindy_sketch(scenario)
    shape_env = ℝenv()
    comp_env = ComponentEnv()
    components_scalar_arithmatic!(comp_env, can_grow=true)

    basis_expr = TAST[]
    basis_weights = Float64[]
    for v1 in sketch.input_vars
        push!(basis_expr, v1)
        push!(basis_weights, 25.0)
        for v2 in sketch.input_vars
            if v2.name <= v1.name
                push!(basis_expr, v1*v2)
                push!(basis_weights, 200.0)
            end
        end
    end
    @show basis_expr
    basis = [compile(e, shape_env, comp_env) for e in basis_expr]
    regressor = LassoRegression(1.0; fit_intercept=true)
    # regressor = RidgeRegression(1.0; fit_intercept=true)
    # regressor = LinearRegression(fit_intercept=true)
    optimizer = SeqThresholdOptimizer(0.1, regressor)
    SindySynthesis(comp_env, basis, sketch, optimizer)
end

em_result = let
    post_sampler = ParticleFilterSampler(
        n_particles=60_000,
        n_trajs=100,
    )
    comps_σ = [0.1,0.1,0.1]
    comps_guess = OrderedDict(
        :loc_ax => GaussianComponent(_ -> 0.0, 0.1),
        :loc_ay => GaussianComponent(_ -> 0.0, 0.1),
        :der_ω => GaussianComponent(_ -> 0.0, 0.1),
    )
    # comps_guess = true_comps

    test_scenario(scenario, sim_result, algorithm, comps_σ, post_sampler, n_fit_trajs)
    synthesize_scenario(
        scenario, sim_result, algorithm, comps_guess; post_sampler, n_fit_trajs)
end
nothing
##-----------------------------------------------------------
# plot and save the results
show_dyn_history(em_result.iter_result.dyn_history, table_width=50)
summary_dir=joinpath(save_dir, "summary") |> mkdir
open(joinpath(summary_dir, "iter_result.txt"), "w") do io
    println(io, em_result)
end
# let param_plt = plot_params(em_result)
#     display(param_plt)
#     savefig(param_plt, joinpath(summary_dir, "params.svg"))
# end
typeof(em_result.iter_result)
let perf_plot = plot_performance(em_result, start_idx=10)
    display(perf_plot)
    savefig(perf_plot, joinpath(summary_dir, "performance.svg"))
end
##-----------------------------------------------------------

