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
using Alert
using Random
using Statistics

!true && begin
    include("../src/SEDL.jl")
    using .SEDL
end
using SEDL
using .SEDL: @kwdef

include("data_from_source.jl")
##-----------------------------------------------------------
# set up scenario and simulate some data
if !@isdefined is_quick_test
    is_quick_test = false
end
is_quick_test && @info "Quick testing VI..."

Random.seed!(123)

should_train_dynamics = true  # whether to train a NN motion model or use the ground truth
use_gpu = true
use_simple_obs_model = true
tconf = SEDL.TensorConfig(use_gpu, Float32)
data_source = RealData(
    SEDL.data_dir("real_data", "simple_loop"),
    SEDL.data_dir("real_data", "simple_loop_test"),
)
# data_source = SimulationData(; n_train_ex=16, n_test_ex=64, times=0:tconf(0.1):10)
train_method = :EM  # :VI or :EM or :Supervised
device = use_gpu ? Flux.gpu : Flux.cpu
use_sim_data = data_source isa SimulationData

landmarks = [[10.0, 0.0], [-4.0, -2.0], [-6.0, 5.0]]
landmarks_tensor = landmarks |> SEDL.hcatreduce |> x -> Flux.cat(x'; dims=3) |> device
@smart_assert size(landmarks_tensor) == (length(landmarks), 2, 1)

linfo = SEDL.LandmarkInfo(; landmarks)
sce = if use_sim_data
    SEDL.HovercraftScenario()
    # sce = SEDL.Car2dScenario(linfo, SEDL.BicycleCarDyn(; front_drive=false))
else
    SEDL.RealCarScenario()
end
state_L2_loss = SEDL.state_L2_loss_batched(sce)

sketch = SEDL.batched_sketch(sce)
obs_model = if use_simple_obs_model
    let σs = use_sim_data ? (pos=0.1, θ=5°) : (pos=0.1, angle_2d=5°)
        state -> SEDL.gaussian_obs_model(state, σs)
    end
else
    # TODO: fix this for angle_2d
    state -> SEDL.landmark_obs_model(
        state, (; landmarks=landmarks_tensor, σ_bearing=15°, σ_range=5.0)
    )
end

data_train, data_test = if data_source isa SimulationData
    true_params = map(p -> convert(tconf.ftype, p), SEDL.simulation_params(sce))
    motion_model = SEDL.BatchedMotionModel(
        tconf, sketch, SEDL.batched_core(sce, true_params)
    )
    data_from_source(sce, data_source, tconf; motion_model, obs_model)
else
    data_from_source(sce, data_source, tconf; obs_model)
end

n_train_ex = data_train.states[1].batch_size
obs_frames = eachindex(data_train.times)

let
    visual_id = 1
    first_states = [map(m -> m[:], Flux.cpu(b[visual_id].val)) for b in data_train.states]
    landmark_obs = [((; landmarks=fill(true, length(landmarks)))) for _ in first_states]
    obs_data = (; obs_frames=1:10:length(data_train.times), observations=landmark_obs)
    plot()
    SEDL.plot_2d_scenario!(first_states, obs_data, "Ground truth"; landmarks)
    SEDL.plot_2d_trajectories!(data_train.states, "forward simulation") |> display
end

SEDL.plot_batched_series(data_train.times, getindex.(data_train.states, 1)) |> display
##-----------------------------------------------------------
# test particle filtering
function plot_pf_posterior(
    motion_model,
    sim_result,
    sample_id;
    n_particles=100_000,
    obs_frames=nothing,
    plot_args...,
)
    isnothing(obs_frames) && (obs_frames = eachindex(sim_result.times))
    pf_result = SEDL.batched_particle_filter(
        repeat(sim_result.states[1][sample_id], n_particles),
        (;
            sim_result.times,
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
        sim_result.times,
        pf_trajs;
        truth=getindex.(sim_result.states, sample_id),
        plot_args...,
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
    guide_trajs =
        guide(
            repeat(sim_result.states[1][sample_id], n_particles),
            getindex.(sim_result.observations, sample_id),
            getindex.(sim_result.controls, sample_id),
            sim_result.Δt,
        ).trajectory

    SEDL.plot_batched_series(
        sim_result.times,
        guide_trajs;
        truth=getindex.(sim_result.states, sample_id),
        plot_args...,
    )
end

function posterior_metrics(
    motion_model, data; n_repeats=1, obs_frames=nothing, n_particles=100_000
)
    rows = map(1:n_repeats) do _
        SEDL.estimate_posterior_quality(
            motion_model, obs_model, data; n_particles, obs_frames, state_L2_loss
        )
    end
    if n_repeats == 1
        rows[1]
    else
        SEDL.named_tuple_reduce(rows, SEDL.to_measurement)
    end
end


function with_alert(task::Function, task_name::String)
    try
        task()
        alert("$task_name finished.")
    catch e
        alert("$task_name stopped due to exception: $e.")
        rethrow()
    end
end

if use_sim_data
    plot_pf_posterior(motion_model, data_train, 1; title="true posterior") |> display
end
##-----------------------------------------------------------
# set up the NN models
h_dim = 64
y_dim = sum(m -> size(m, 1), data_train.observations[1].val)
@info "Computing normal transforms..."
normal_transforms = @time SEDL.compute_normal_transforms(
    sketch, data_train.states, data_train.controls, data_train.observations, data_train.Δt
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
    learned_motion_model,
    data_train,
    1;
    obs_frames=1:2,
    title="Motion model prior (initial)",
) |> display

if train_method == :VI
    guide =
        SEDL.mk_guide(;
            sketch, dynamics_core=learned_motion_model.core, h_dim, y_dim, normal_transforms
        ) |> device

    let (prior_trajs, _) = guide(
            data_train.states[1],
            data_train.observations,
            data_train.controls,
            data_train.Δt,
        )
        SEDL.plot_batched_series(
            data_train.times,
            SEDL.TensorConfig(false).(prior_trajs);
            title="Guide posterior (initial)",
        ) |> display
    end
end

# adam = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.WeightDecay(1e-4), Flux.ADAM(1e-4))
adam = Flux.ADAM(1e-4)
settings = (;
    scenario=summary(sce),
    use_sim_data,
    train_method,
    h_dim,
    should_train_dynamics,
    use_simple_obs_model,
    n_train_ex,
)
save_dir = SEDL.data_dir(savename("test_vi", settings; connector="-"))
if isdir(save_dir)
    @warn "removing old data at $save_dir..."
    rm(save_dir; recursive=true)
    @tagsave(joinpath(save_dir, "settings.bson"), @dict(settings))
end

logger = TBLogger(joinpath(save_dir, "tb_logs"))
@info """To view tensorboard logs, use the following command: 
```
tensorboard --samples_per_plugin "images=100" --logdir "$save_dir/tb_logs"
```"""
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
            test_scores = posterior_metrics(learned_motion_model, data_test)

            Base.with_logger(logger) do
                @info "testing" log_obs = test_scores.mean log_step_increment = 0
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? data_train : data_test
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
                # display(plt)
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
    with_alert("EM training") do
        total_steps = 10_000
        n_steps = is_quick_test ? 3 : total_steps + 1
        @info "Training the dynamics using EM"
        SEDL.train_dynamics_em!(
            learned_motion_model,
            obs_model,
            data_train.states[1],
            data_train.observations,
            data_train.controls,
            (; data_train.times, obs_frames);
            optimizer=adam,
            n_steps,
            trajs_per_step=1,
            # sampling_model=motion_model,
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
# train the model using supervised learning (assuming having ground truth trajectories)
function supervised_callback(
    learned_motion_model::BatchedMotionModel; n_steps, test_every=100, plot_every=10_000
)
    prog = Progress(n_steps; showspeed=true)
    test_in_set, test_out_set = SEDL.input_output_from_trajectory(
        learned_motion_model.sketch, data_test.states, data_test.controls, data_test.times
    )
    test_input = BatchTuple(test_in_set)
    test_output = BatchTuple(test_out_set)
    test_loss() =
        -SEDL.transition_logp(learned_motion_model.core, test_input, test_output) /
        test_input.batch_size

    function (r)
        Base.with_logger(logger) do
            @info "training" r.loss r.lr
            @info "statistics" r.time_stats... log_step_increment = 0
        end

        # Compute test log_obs and plot a few trajectories.
        if r.step % test_every == 1
            Base.with_logger(logger) do
                @info("testing", test_loss = test_loss(), log_step_increment = 0)
            end
        end
        if r.step % plot_every == 1
            test_scores = posterior_metrics(learned_motion_model, data_test)

            Base.with_logger(logger) do
                @info("testing", log_obs = test_scores.mean, log_step_increment = 0)
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? data_train : data_test
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
            end
        end

        next!(prog; showvalues=[(:loss, r.loss), (:step, r.step), (:learning_rate, r.lr)])
    end
end

(train_method == :Supervised) && let
    use_ground_truth = true
    if use_ground_truth
        core_in_set, core_out_set = SEDL.input_output_from_trajectory(
            learned_motion_model.sketch,
            data_train.states,
            data_train.controls,
            data_train.times;
            test_consistency=true,
        )
    else
        core_in_set, core_out_set = BatchTuple[], BatchTuple[]
        x0_batch = data_train.states[1]
        times = data_train.times
        @showprogress for ex_id in 1:(x0_batch.batch_size)
            x0 = repeat(x0_batch[ex_id], 100_000)
            controls = getindex.(data_train.controls, ex_id)
            observations = getindex.(data_train.observations, ex_id)
            local pf_result = SEDL.batched_particle_filter(
                x0,
                (; times, obs_frames, controls, observations);
                motion_model,
                obs_model,
                record_io=true,
                showprogress=false,
            )
            (; core_input_seq, core_output_seq) = SEDL.batched_trajectories(
                pf_result, 1; record_io=true
            )
            append!(core_in_set, core_input_seq)
            append!(core_out_set, core_output_seq)
        end
    end
    @info "Number of training data: $(sum(x -> x.batch_size, core_in_set))"

    with_alert("Supervised training") do
        total_steps = 50_000
        n_steps = is_quick_test ? 3 : total_steps + 1
        SEDL.train_dynamics_supervised!(
            learned_motion_model.core,
            BatchTuple(core_in_set),
            BatchTuple(core_out_set);
            optimizer=adam,
            n_steps,
            callback=supervised_callback(learned_motion_model; n_steps),
        )
    end
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

        test_x0_batch = data_test.states[1]
        # Compute test elbo and plot a few trajectories.
        if r.step % test_every == 1
            test_trajs, test_lp_guide, test_core_in, test_core_out = guide(
                test_x0_batch, data_test.observations, data_test.controls, data_test.Δt
            )
            test_lp_dynamics = SEDL.transition_logp(
                learned_motion_model.core, test_core_in, test_core_out
            )
            test_lp_obs = SEDL.observation_logp(
                obs_model, test_trajs, data_test.observations
            )
            test_elbo =
                (test_lp_dynamics + test_lp_obs - sum(test_lp_guide)) /
                (length(test_trajs) * test_x0_batch.batch_size)

            Base.with_logger(logger) do
                @info "testing" elbo = test_elbo log_step_increment = 0
            end

            test_scores = posterior_metrics(learned_motion_model, data_test)

            Base.with_logger(logger) do
                @info "testing" log_obs = test_scores.mean log_step_increment = 0
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? data_train : data_test
                pf_plt = plot_pf_posterior(
                    learned_motion_model,
                    sim_data,
                    id;
                    title="$name traj $id (iter $(r.step))",
                )

                guide_plt = plot_guide_posterior(
                    guide, sim_data, id; title="$name traj $id (iter $(r.step))"
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
    with_alert("VI training") do
        total_steps = 6_000
        n_steps = is_quick_test ? 3 : total_steps + 1

        @info "Training the guide..."
        train_result = @time SEDL.train_guide!(
            guide,
            learned_motion_model.core,
            obs_model,
            data_train.states[1],
            data_train.observations,
            data_train.controls,
            data_train.Δt;
            optimizer=adam,
            n_steps,
            anneal_schedule=step -> linear(1e-3, 1.0)(min(1, 1.5step / n_steps)),
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
plot_pf_posterior(
    learned_motion_model,
    data_train,
    1;
    obs_frames=1:1,
    title="Open-loop prediction (learned)",
) |> display
if use_sim_data
    plot_pf_posterior(
        motion_model,
        data_train,
        1;
        obs_frames=1:1,
        title="Open-loop prediction (true model)",
    ) |> display
end

function evaluate_model(motion_model, data_test, name)
    metrics = posterior_metrics(motion_model, data_test, n_repeats=5)
    open_loop = posterior_metrics(motion_model, data_test, n_repeats=5, obs_frames=1:1).RMSE
    (; name, metrics..., open_loop)
end

perf_table = let
    @info "Testing dynamics model performance on the test set"
    rows = [
        evaluate_model(learned_motion_model, data_test, "EM"),
    ]
    if use_sim_data
        push!(rows, evaluate_model(motion_model, data_test, "true_model"))
    end
    DataFrame(rows)
end
wsave(joinpath(save_dir, "logp_obs_table.bson"), @dict perf_table)
display(perf_table)
##-----------------------------------------------------------
run(`ls -lh "$save_dir"/tb_logs`)
save_dir