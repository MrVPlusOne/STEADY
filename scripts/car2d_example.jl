##-----------------------------------------------------------
using Distributions
using StatsPlots
using DrWatson
import Random

StatsPlots.default(dpi=300, legend=:outerbottom)

# generate data
Random.seed!(123)
landmarks = @SVector[@SVector[-1.0, 2.5], @SVector[6.0, -4.0], @SVector[10.0, 2.0]]
lInfo = LandmarkInfo(; landmarks, bearing_only=Val(true))
scenario = Car2dScenario(lInfo, SimpleSlidingCar())

times = collect(0.0:0.1:14)
obs_frames = 1:5:length(times)

true_params = (; 
    mass=2.0, drag_x=0.05, drag_y=0.11, rot_mass=0.65, rot_drag=0.07,
    sep=0.48, len=0.42, fraction_max=1.5, σ_v=0.011, σ_ω=0.008,)
params_guess = nothing

function manual_control()
    pert(x) = x + 0.01randn()
    @unzip times, v̂_seq, steer_seq = [
        (t=0.0, v̂=0.0, steer=0.0),
        (t=1.0, v̂=pert(3.2), steer=pert(10°)),
        (t=4.0, v̂=pert(3.0), steer=pert(10°)),
        (t=4.5, v̂=pert(3.5), steer=pert(-30°)),
        (t=6.5, v̂=pert(3.3), steer=pert(-30°)),
        (t=7.2, v̂=pert(3.0), steer=pert(20°)),
        (t=9.0, v̂=pert(2.8), steer=pert(20°)),
        (t=9.6, v̂=pert(2.5), steer=pert(10°)),
        (t=15.0, v̂=2.0, steer=0.0),
    ]
    if rand() < 0.6
        steer_seq = -steer_seq
    end
    v̂_f = LinearInterpolation(times, v̂_seq)
    steer_f = LinearInterpolation(times, steer_seq)
    (s, obs, t::Float64) -> begin
        (v̂ = v̂_f(t), steer = steer_f(t))
    end
end

n_runs = 6
n_fit_trajs = 15
setups = map(1:n_runs) do i
    x0 = (
        pos=@SVector[-6.5+randn(), 1.2+randn()], 
        vel=@SVector[0.25, 0.0],
        θ=randn()°, 
        ω=0.1randn(),
    )
    ScenarioSetup(times, obs_frames, x0, manual_control())
end

comp_env = ComponentEnv()
components_scalar_arithmatic!(comp_env, can_grow=true)
components_special_functions!(comp_env, can_grow=true)

NamedTuple()
comps_guess = let 
    vars = variables(scenario)
    (; loc_vx, loc_vy, ω, steer) = vars
    (; fraction_max, v̂, mass, drag_x, drag_y, rot_mass, rot_drag, len) = vars
    f_zero = get_component(comp_env, :zero)
    (f_x=fraction_max, f_y=fraction_max, f_θ = fraction_max * len,)
end
##-----------------------------------------------------------
# run the scenario
save_dir=datadir("sims/car2d")
scenario_result = run_scenario(scenario, true_params, setups; 
    save_dir, comp_env, comps_guess, params_guess, n_fit_trajs, 
    max_iters=1)
iter_result = scenario_result.iter_result
display(iter_result)
##-----------------------------------------------------------