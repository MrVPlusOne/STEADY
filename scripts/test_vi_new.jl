##-----------------------------------------------------------
# imports
using Flux: Flux
using CUDA: CUDA
CUDA.allowscalar(false)
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)

include("../src/SEDL.jl")  # reloads the module
using .SEDL: SEDL
using .SEDL: @smart_assert
##---------------------------------------------------------- 
# set up scenario and simulate some data
is_quick_test = false
use_gpu = true
tconf = SEDL.TensorConfig(use_gpu, Float32)
device = use_gpu ? Flux.gpu : Flux.cpu

landmarks = [[10.0, 0.0], [-4.0, -2.0], [-6.0, 5.0]]
landmarks_tensor = landmarks |> SEDL.hcatreduce |> x -> Flux.cat(x'; dims=3) |> device
@smart_assert size(landmarks_tensor) == (length(landmarks), 2, 1)
size(landmarks_tensor)

linfo = SEDL.LandmarkInfo(; landmarks)
sce = SEDL.HovercraftScenario(linfo)

n_trajs = 20
x0_batch = SEDL.BatchTuple(
    tconf,
    [(;
        pos=[0.5, 1.0] + 2randn(2),
        vel=[0.25 + 0.3randn(), 0.1 + 0.2randn()],
        θ=[π / 5 + π * randn()],
        ω=[π / 50 + 0.1π * randn()],
    ) for _ in 1:n_trajs],
)


times = 0:0.1:10
obs_frames = 1:10:length(times)

true_params = map(p -> convert(tconf.ftype, p), SEDL.simulation_params(sce))
motion_model = SEDL.BatchedMotionModel(
    tconf, SEDL.batched_sketch(sce), SEDL.batched_core(sce, true_params)
)

obs_model =
    state ->
        SEDL.landmark_obs_sample(state, (; landmarks=landmarks_tensor, linfo.σ_bearing))

controller = let scontroller = SEDL.simulation_controller(sce)
    (args...) -> begin
        (; ul, ur) = scontroller(args...)
        SEDL.BatchTuple(tconf, [(ul=[ul], ur=[ur])])
    end
end


sim_en = SEDL.simulate_trajectory(
    times, x0_batch, (; motion_model=(x -> x.new_state) ∘ motion_model, obs_model), controller
)

let
    first_states = [Flux.cpu(b[1].val) for b in sim_en.states]
    landmark_obs = [((; landmarks=fill(true, length(landmarks)))) for _ in first_states]
    obs_data = (; obs_frames, observations=landmark_obs)
    plot()
    SEDL.plot_2d_scenario!(first_states, obs_data, "Ground truth"; landmarks)
    SEDL.plot_2d_trajectories!(sim_en.states, "forward simulation") |> display
end

SEDL.plot_batched_series(times, SEDL.TensorConfig(false).(sim_en.states))
##-----------------------------------------------------------


