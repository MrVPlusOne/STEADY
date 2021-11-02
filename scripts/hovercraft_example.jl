##-----------------------------------------------------------
using Distributions
using StatsPlots
using DrWatson
import Random

StatsPlots.default(dpi=600, legend=:outerbottom)
##-----------------------------------------------------------
# generate data
landmarks = @SVector[
    @SVector[-1.0, 2.5], @SVector[0.5, -1.0],
    @SVector[7.0, -6.0], @SVector[13.0, -7.5]]
scenario = HovercraftScenario(;landmarks)
times = collect(0.0:0.1:12)
obs_frames = 1:5:length(times)
true_params = (; 
    mass = 1.5, drag_x = 0.02, drag_y=0.05, rot_mass=1.5, rot_drag=0.07, sep=0.81,
    σ_v=0.04, σ_ω=0.03)
params_guess = (; 
    mass = 1.65, drag_x = 0.025, drag_y=0.056, rot_mass=1.66, rot_drag=0.09, sep=0.87,
    σ_v=0.1, σ_ω=0.1)
x0 = (pos=@SVector[0.5, 0.5], vel=@SVector[0.25, 0.0], θ=0°, ω=0.0)

function manual_control()
    @unzip times, ul_seq, ur_seq = [
        (t=0.0, ul=0.0, ur=0.0),
        (t=0.5, ul=1.0, ur=0.4),
        (t=2.0, ul=0.0, ur=0.0),
        (t=3.0, ul=0.5, ur=0.5),
        (t=5.0, ul=1.1, ur=0.5),        
        (t=6.0, ul=0.0, ur=0.0),        
        (t=9.0, ul=0.5, ur=1.0),   
        (t=11.0, ul=0.0, ur=0.4),     
        (t=12.0, ul=0.0, ur=0.0),     
        (t=15.0, ul=0.0, ur=0.0),
    ]
    ul_f = LinearInterpolation(times, ul_seq)
    ur_f = LinearInterpolation(times, ur_seq)
    (s, obs, t::Float64) -> begin
        (ul = ul_f(t), ur = ur_f(t))
    end
end

comp_env = ComponentEnv()
components_scalar_arithmatic!(comp_env, can_grow=true)
# components_special_functions!(comp_env, can_grow=true)

comps_guess = let 
    vars = variables(scenario)
    (; ul, mass, rot_mass, sep) = vars
    f_zero = get_component(comp_env, :zero)
    (f_x=f_zero(ul), f_y=f_zero(ul), f_θ = f_zero(ul) * sep,)
end
##-----------------------------------------------------------
Random.seed!(123)
scenario_result = run_scenario(scenario, true_params, x0, manual_control(); 
    times, obs_frames, comp_env, comps_guess, params_guess, max_iters=150)

iter_result = scenario_result.iter_result
display(iter_result)
plot_params(iter_result) |> display
plot(iter_result, start_idx=1) |> display
##-----------------------------------------------------------

