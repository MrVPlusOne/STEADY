##-----------------------------------------------------------
# imports
using DrWatson
using Revise
using Flux: Flux
using CUDA: CUDA, CuArray
CUDA.allowscalar(false)
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)
using TensorBoardLogger: TBLogger
using ProgressMeter
using Serialization: serialize, deserialize
using DataFrames: DataFrame

!true && begin
    include("../src/SEDL.jl")
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
use_simple_obs_model = true
train_method = :VI  # :VI or :EM
tconf = SEDL.TensorConfig(use_gpu, Float32)
device = use_gpu ? Flux.gpu : Flux.cpu

landmarks = [[10.0, 0.0], [-4.0, -2.0], [-6.0, 5.0]]
landmarks_tensor = landmarks |> SEDL.hcatreduce |> x -> Flux.cat(x'; dims=3) |> device
@smart_assert size(landmarks_tensor) == (length(landmarks), 2, 1)
size(landmarks_tensor)

linfo = SEDL.LandmarkInfo(; landmarks)
# sce = SEDL.HovercraftScenario(linfo)
sce = SEDL.Car2dScenario(linfo, SEDL.BicycleCarDyn(; front_drive=false))

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

obs_model = if use_simple_obs_model
    state -> SEDL.gaussian_obs_model(state, (pos=1.0, vel=1.0, θ=1.0, ω=1.0))
else
    state -> SEDL.landmark_obs_model(
        state, (; landmarks=landmarks_tensor, σ_bearing=15°, σ_range=2.0)
    )
end

controller = let scontroller = SEDL.simulation_controller(sce)
    (args...) -> begin
        control = scontroller(args...)
        SEDL.BatchTuple(tconf, [map(v -> [v], control)])
    end
end

x0_batch = SEDL.BatchTuple(tconf, [sample_x0() for _ in 1:128])
test_x0_batch = SEDL.BatchTuple(tconf, [sample_x0() for _ in 1:64])

sample_next_state = (x -> x.next_state) ∘ motion_model
sample_observation = rand ∘ obs_model
sim_en = SEDL.simulate_trajectory(
    times, x0_batch, sample_next_state, sample_observation, controller
)
test_sim = SEDL.simulate_trajectory(
    times, test_x0_batch, sample_next_state, sample_observation, controller
)

let
    visual_id = 5
    first_states = [Flux.cpu(b[visual_id].val) for b in sim_en.states]
    landmark_obs = [((; landmarks=fill(true, length(landmarks)))) for _ in first_states]
    obs_data = (; obs_frames, observations=landmark_obs)
    plot()
    SEDL.plot_2d_scenario!(first_states, obs_data, "Ground truth"; landmarks)
    SEDL.plot_2d_trajectories!(sim_en.states, "forward simulation") |> display
end

SEDL.plot_batched_series(times, sim_en.states) |> display
##-----------------------------------------------------------
# test particle filtering
function plot_pf_posterior(
    motion_model,
    sim_result,
    sample_id;
    n_particles=100_000,
    obs_frames=obs_frames,
    plot_args...,
)
    pf_result = SEDL.batched_particle_filter(
        repeat(sim_result.states[1][sample_id], n_particles),
        (;
            times,
            obs_frames,
            controls=getindex.(sim_result.controls, sample_id),
            observations=getindex.(sim_result.observations, sample_id),
        );
        motion_model,
        obs_model,
        showprogress=false,
    )
    pf_trajs = SEDL.batched_trajectories(pf_result, 100)
    SEDL.plot_batched_series(
        times, pf_trajs; truth=getindex.(sim_result.states, sample_id), plot_args...
    )
end

function plot_guide_posterior(
    guide::VIGuide,
    sim_result,
    sample_id;
    n_particles=100,
    obs_frames=obs_frames,
    plot_args...,
)
    guide_trajs = guide(
        repeat(sim_result.states[1][sample_id], n_particles),
        getindex.(sim_result.observations, sample_id),
        getindex.(sim_result.controls, sample_id),
        Δt,
    ).trajectory

    SEDL.plot_batched_series(
        times, guide_trajs; truth=getindex.(sim_result.states, sample_id), plot_args...
    )
end

function estimate_logp_pf(
    motion_model, sim_result; n_particles=100_000, obs_frames=obs_frames
)
    n_ex = sim_result.states[1].batch_size
    log_obs_values = @showprogress 0.1 "estimate_logp_pf" map(1:n_ex) do sample_id
        pf_result = SEDL.batched_particle_filter(
            repeat(sim_result.states[1][sample_id], n_particles),
            (;
                times,
                obs_frames,
                controls=getindex.(sim_result.controls, sample_id),
                observations=getindex.(sim_result.observations, sample_id),
            );
            motion_model,
            showprogress=false,
            obs_model,
        )
        pf_result.log_obs
    end
    (;
        mean=mean(log_obs_values),
        std=std(log_obs_values),
        max=maximum(log_obs_values),
        min=minimum(log_obs_values),
    )
end

plot_pf_posterior(motion_model, sim_en, 1; title="true posterior") |> display
##-----------------------------------------------------------
# set up the NN models
h_dim = 64
y_dim = sum(m -> size(m, 1), sim_en.observations[1].val)
normal_transforms = @time SEDL.compute_normal_transforms(
    sketch, sim_en.states, sim_en.controls, sim_en.observations, Δt
)

learned_motion_model = if should_train_dynamics
    nn_motion_model =
        SEDL.mk_nn_motion_model(; sketch, tconf, h_dim, normal_transforms) |> device
    @smart_assert !isempty(Flux.params(nn_motion_model))
    nn_motion_model
else
    motion_model
end

plot_pf_posterior(
    learned_motion_model, sim_en, 1; obs_frames=1:2, title="Motion model prior (initial)"
) |> display

# adam = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.WeightDecay(1e-4), Flux.ADAM(1e-4))
adam = Flux.ADAM(1e-4)
settings = (;
    scenario=summary(sce), train_method, h_dim, should_train_dynamics, use_simple_obs_model
)
save_dir = SEDL.data_dir(savename("test_vi", settings; connector="-"))
if isdir(save_dir)
    @warn "removing old data at $save_dir..."
    rm(save_dir; recursive=true)
    @tagsave(joinpath(save_dir, "settings.bson"), @dict(settings))
end

logger = TBLogger(joinpath(save_dir, "tb_logs"))
@info """To view tensorboard logs, use `tensorboard --host 0.0.0.0 --samples_per_plugin "images=100" --logdir "$save_dir/tb_logs"`"""

if train_method == :VI
    guide =
        SEDL.mk_guide(;
            sketch, dynamics_core=learned_motion_model.core, h_dim, y_dim, normal_transforms
        ) |> device

    let (prior_trajs, _) = guide(sim_en.states[1], sim_en.observations, sim_en.controls, Δt)
        SEDL.plot_batched_series(
            times, SEDL.TensorConfig(false).(prior_trajs); title="Guide posterior (initial)"
        ) |> display
    end
end
##-----------------------------------------------------------
# train the model using expectation maximization

function em_callback(learned_motion_model::BatchedMotionModel; n_steps, test_every=100)
    prog = Progress(n_steps; showspeed=true)
    function (r)
        Base.with_logger(logger) do
            @info "training" r.log_obs r.loss r.lr
            @info "statistics" r.time_stats... log_step_increment = 0
        end

        # Compute test log_obs and plot a few trajectories.
        if r.step % test_every == 1
            test_scores = estimate_logp_pf(learned_motion_model, test_sim)

            Base.with_logger(logger) do
                @info "testing" log_obs = test_scores.mean log_step_increment = 0
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? sim_en : test_sim
                plt = plot_pf_posterior(
                    learned_motion_model,
                    sim_data,
                    id;
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
                (:log_obs, r.log_obs),
                (:loss, r.loss),
                (:step, r.step),
                (:learning_rate, r.lr),
            ],
        )
    end
end

if train_method == :EM
    let total_steps = 8_000
        n_steps = is_quick_test ? 3 : total_steps + 1
        @info "Training the dynamics using EM"
        SEDL.train_dynamics_em!(
            learned_motion_model,
            obs_model,
            sim_en.states[1],
            sim_en.observations,
            sim_en.controls,
            (; times, obs_frames);
            optimizer=adam,
            n_steps,
            minibatch=4,
            callback=em_callback(learned_motion_model; n_steps, test_every=500),
            # lr_schedule=let β = 20^2 / total_steps
            #     step -> 1e-3 / sqrt(β * step)
            # end,
        )
    end
    # save the model
    serialize(
        joinpath(save_dir, "models.serial"),
        (learned_motion_model=Flux.cpu(learned_motion_model),),
    )
    deserialize(joinpath(save_dir, "models.serial"))
end
##-----------------------------------------------------------
# train the guide using variational inference
function vi_callback(learned_motion_model::BatchedMotionModel; n_steps, test_every=200)
    prog = Progress(n_steps; showspeed=true)
    function (r)
        Base.with_logger(logger) do
            @info "training" r.elbo r.loss r.lr r.annealing
            @info "statistics" r.time_stats... log_step_increment = 0
        end

        # Compute test elbo and plot a few trajectories.
        if r.step % test_every == 1
            test_trajs, test_lp_guide, test_core_in, test_core_out = guide(
                test_x0_batch, test_sim.observations, test_sim.controls, Δt
            )
            test_lp_dynamics = SEDL.transition_logp(
                learned_motion_model.core, test_core_in, test_core_out
            )
            test_lp_obs = SEDL.observation_logp(
                obs_model, test_trajs, test_sim.observations
            )
            test_elbo =
                (test_lp_dynamics + test_lp_obs - sum(test_lp_guide)) /
                (length(test_trajs) * test_x0_batch.batch_size)

            Base.with_logger(logger) do
                @info "testing" elbo = test_elbo log_step_increment = 0
            end

            test_scores = estimate_logp_pf(learned_motion_model, test_sim)

            Base.with_logger(logger) do
                @info "testing" log_obs = test_scores.mean log_step_increment = 0
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? sim_en : test_sim
                pf_plt = plot_pf_posterior(
                    learned_motion_model,
                    sim_data,
                    id;
                    title="$name traj $id (iter $(r.step))",
                )

                guide_plt = plot_guide_posterior(
                    guide, sim_data, id; title="$name traj $id (iter $(r.step))",
                )

                Base.with_logger(logger) do
                    kv = [Symbol("traj_$id") => guide_plt]
                    @info "$name/guide" log_step_increment = 0 kv...
                    kv = [Symbol("traj_$id") => pf_plt]
                    @info "$name/particle" log_step_increment = 0 kv...
                end
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
end

if train_method == :VI
    let total_steps = 6_000
        n_steps = is_quick_test ? 3 : total_steps + 1

        @info "Training the guide..."
        train_result = @time SEDL.train_guide!(
            guide,
            learned_motion_model.core,
            obs_model,
            sim_en.states[1],
            sim_en.observations,
            sim_en.controls,
            Δt;
            optimizer=adam,
            n_steps,
            anneal_schedule=step -> linear(1e-3, 1.0)(min(1, 3step / n_steps)),
            callback=vi_callback(learned_motion_model; n_steps),
            lr_schedule=let β = 40^2 / total_steps
                step -> 1e-3 / sqrt(β * step)
            end,
        )
        display(train_result)
    end
    # save the model
    serialize(
        joinpath(save_dir, "models.serial"),
        (learned_motion_model=Flux.cpu(learned_motion_model), guide=Flux.cpu(guide)),
    )
    deserialize(joinpath(save_dir, "models.serial"))
end
##-----------------------------------------------------------
let
    @info "Testing dynamics model performance on the test set"
    rows = [
        merge((name="learned",), estimate_logp_pf(learned_motion_model, test_sim)),
        merge((name="truth",), estimate_logp_pf(motion_model, test_sim)),
    ]
    DataFrame(rows)
end
##-----------------------------------------------------------
run(`ls -lh "$save_dir"/tb_logs`)
save_dir