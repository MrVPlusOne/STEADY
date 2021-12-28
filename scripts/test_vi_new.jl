##-----------------------------------------------------------
# imports
using DrWatson
using Revise
using Flux: Flux
using CUDA: CUDA
CUDA.allowscalar(false)
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)
using TensorBoardLogger: TBLogger
using ProgressMeter
using Serialization: serialize, deserialize

!true && begin
    include("../src/SEDL.jl")  # reloads the module
    using .SEDL
end
using SEDL: SEDL
using SEDL
##---------------------------------------------------------- 
# set up scenario and simulate some data
if !@isdefined is_quick_test
    is_quick_test = false
end
is_quick_test && @info "Quick testing VI..."

should_train_dynamics = true  # whether to train a NN motion model or use the ground truth
use_gpu = true
tconf = SEDL.TensorConfig(use_gpu, Float32)
device = use_gpu ? Flux.gpu : Flux.cpu

landmarks = [[10.0, 0.0], [-4.0, -2.0], [-6.0, 5.0]]
landmarks_tensor = landmarks |> SEDL.hcatreduce |> x -> Flux.cat(x'; dims=3) |> device
@smart_assert size(landmarks_tensor) == (length(landmarks), 2, 1)
size(landmarks_tensor)

linfo = SEDL.LandmarkInfo(; landmarks)
sce = SEDL.HovercraftScenario(linfo)

sample_x0() = (;
    pos=[0.5, 1.0] + 2randn(2),
    vel=[0.25 + 0.3randn(), 0.1 + 0.2randn()],
    θ=[π / 5 + π * randn()],
    ω=[π / 50 + 0.1π * randn()],
)

Δt = tconf(0.1)
times = 0:Δt:10
obs_frames = 1:10:length(times)

true_params = map(p -> convert(tconf.ftype, p), SEDL.simulation_params(sce))
sketch = SEDL.batched_sketch(sce)
motion_model = SEDL.BatchedMotionModel(tconf, sketch, SEDL.batched_core(sce, true_params))
@smart_assert isempty(Flux.params(motion_model))

# obs_model =
#     state -> SEDL.landmark_obs_model(state, (; landmarks=landmarks_tensor, linfo.σ_bearing))
obs_model = state -> SEDL.gaussian_obs_model(state, (pos=1.0, vel=1.0, θ=1.0, ω=1.0))

controller = let scontroller = SEDL.simulation_controller(sce)
    (args...) -> begin
        (; ul, ur) = scontroller(args...)
        SEDL.BatchTuple(tconf, [(ul=[ul], ur=[ur])])
    end
end

x0_batch = SEDL.BatchTuple(tconf, [sample_x0() for _ in 1:128])
test_x0_batch = SEDL.BatchTuple(tconf, [sample_x0() for _ in 1:128])

sim_en = SEDL.simulate_trajectory(times, x0_batch, (; motion_model, obs_model), controller)
test_sim = SEDL.simulate_trajectory(
    times, test_x0_batch, (; motion_model, obs_model), controller
)
# repeat_factor = 128
# sim_en = map(sim_en) do seq
#     map(b -> repeat(b, repeat_factor), seq)
# end

let
    first_states = [Flux.cpu(b[1].val) for b in sim_en.states]
    landmark_obs = [((; landmarks=fill(true, length(landmarks)))) for _ in first_states]
    obs_data = (; obs_frames, observations=landmark_obs)
    plot()
    SEDL.plot_2d_scenario!(first_states, obs_data, "Ground truth"; landmarks)
    SEDL.plot_2d_trajectories!(sim_en.states, "forward simulation") |> display
end

SEDL.plot_batched_series(times, TensorConfig(false).(sim_en.states)) |> display
##-----------------------------------------------------------
# set up the NN models
h_dim = 128
y_dim = sum(m -> size(m, 1), sim_en.observations[1].val)
normal_transforms = SEDL.compute_normal_transforms(
    sketch, sim_en.states, sim_en.controls, sim_en.observations, Δt
)
guide = @time SEDL.mk_guide(; sketch, h_dim, y_dim, normal_transforms) |> device

vi_motion_model = if should_train_dynamics
    nn_motion_model = SEDL.mk_nn_motion_model(; sketch, tconf, h_dim, normal_transforms)
    @smart_assert !isempty(Flux.params(nn_motion_model))
    nn_motion_model
else
    motion_model
end

# adam = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.WeightDecay(1e-4), Flux.ADAM(1e-4))
adam = Flux.ADAM(1e-4)
save_dir = SEDL.data_dir(
    savename("test_vi", (; h_dim, should_train_dynamics); connector="-")
)
if isdir(save_dir)
    @warn "removing old data at $save_dir..."
    rm(save_dir; recursive=true)
end
logger = TBLogger(joinpath(save_dir, "tb_logs"))

elbo_history = []
let (prior_trajs, _) = guide(sim_en.states[1], sim_en.observations, sim_en.controls, Δt)
    SEDL.plot_batched_series(
        times, SEDL.TensorConfig(false).(prior_trajs); title="Guide prior"
    ) |> display
end
##-----------------------------------------------------------
# train the guide
linear(from, to) = x -> from + (to - from) * x

total_steps = 10_000
let n_steps = is_quick_test ? 3 : total_steps + 1, prog = Progress(n_steps; showspeed=true)
    n_visual_trajs = 100

    callback(r) = begin
        push!(elbo_history, r.elbo)
        Base.with_logger(logger) do
            @info "training" r.elbo r.loss r.lr r.annealing
        end

        # Compute test elbo and plot a few trajectories.
        if r.step % 100 == 1
            test_trajs, test_lp_guide = guide(
                test_x0_batch, test_sim.observations, test_sim.controls, Δt
            )
            test_lp_dynamics = SEDL.transition_logp(
                vi_motion_model, test_trajs, test_sim.controls, Δt
            )
            test_lp_obs = SEDL.observation_logp(
                obs_model, test_trajs, test_sim.observations
            )
            test_elbo =
                sum(test_lp_dynamics + test_lp_obs - test_lp_guide) /
                (length(test_trajs) * test_x0_batch.batch_size)

            Base.with_logger(logger) do
                @info "testing" elbo = test_elbo log_step_increment = 0
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? sim_en : test_sim
                visual_states = map(b -> b[id], sim_data.states)
                visual_obs = map(b -> repeat(b[id], n_visual_trajs), sim_data.observations)
                visual_controls = map(b -> repeat(b[id], n_visual_trajs), sim_data.controls)
                test_trajs, _ = guide(
                    repeat(visual_states[1], n_visual_trajs),
                    visual_obs,
                    visual_controls,
                    Δt,
                )
                plt = SEDL.plot_batched_series(
                    times,
                    SEDL.TensorConfig(false).(test_trajs);
                    truth=visual_states,
                    title="$name traj $id (iter $(r.step))",
                )
                Base.with_logger(logger) do
                    kv = [Symbol("traj_$id") => plt]
                    @info name log_step_increment = 0 kv...
                end
                display(plt)
            end
        end

        next!(
            prog;
            showvalues=[
                (:elbo, r.elbo),
                (:step, r.step),
                (:batch_size, r.batch_size),
                (:annealing, r.annealing),
                (:learning_rate, r.lr),
            ],
        )
    end

    @info "Training the guide..."
    train_result = @time SEDL.train_guide!(
        guide,
        vi_motion_model,
        obs_model,
        sim_en.states[1],
        sim_en.observations,
        sim_en.controls,
        Δt;
        optimizer=adam,
        n_steps,
        anneal_schedule=step -> linear(1e-3, 1.0)(min(1, 4step / n_steps)),
        callback,
        lr_schedule=let β = 100^2 / total_steps
            step -> 2e-3 / sqrt(β * step)
        end,
    )
    display(train_result)
end
##-----------------------------------------------------------
plot(elbo_history[500:end]; title="ELBO (training)") |> display
serialize(
    joinpath(save_dir, "models.serial"),
    (vi_motion_model=Flux.cpu(vi_motion_model), guide=Flux.cpu(guide)),
)
deserialize(joinpath(save_dir, "models.serial"))

run(`ls -lh $save_dir/`)
##-----------------------------------------------------------
