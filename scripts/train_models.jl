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
using Setfield
using BSON: BSON

!true && begin
    include("../src/SEDL.jl")
    using .SEDL
end
using SEDL
using .SEDL: @kwdef

include("data_from_source.jl")
##-----------------------------------------------------------
# set up scenario and simulate some data

if !isdefined(Main, :script_args)
    # script_args can be used to override the default config parameters.
    script_args = NamedTuple()
end

(;
    is_quick_test,
    load_trained,
    should_train_dynamics,
    gpu_id,
    use_simple_obs_model,
    use_sim_data,
    train_method,
) = let
    config = (;
        is_quick_test=false,
        load_trained=false,
        should_train_dynamics=true, # whether to train a NN motion model or use the ground truth
        gpu_id=nothing, # integer or nothing
        use_simple_obs_model=true,
        use_sim_data=true,
        train_method=:EM, # :VI or :EM or :Supervised
    )
    merge(config, script_args::NamedTuple)
end
is_quick_test && @info "Quick testing VI..."

Random.seed!(123)

use_gpu = gpu_id !== nothing
if use_gpu
    CUDA.device!(gpu_id)
    @info "Using GPU #$(gpu_id)"
else
    @warn "No GPU specified, using CPU."
end
tconf = SEDL.TensorConfig(use_gpu, Float32)
device = use_gpu ? Flux.gpu : Flux.cpu

data_source = if use_sim_data
    SimulationData(; n_train_ex=16, n_test_ex=64, times=0:tconf(0.1):10)
else
    RealData(
        SEDL.data_dir("real_data", "simple_loop"),
        SEDL.data_dir("real_data", "simple_loop_test"),
    )
end

sce = if use_sim_data
    SEDL.HovercraftScenario()
    # sce = SEDL.Car2dScenario(linfo, SEDL.BicycleCarDyn(; front_drive=false))
else
    SEDL.RealCarScenario()
end
state_L2_loss = SEDL.state_L2_loss_batched(sce)

sketch = SEDL.batched_sketch(sce)

landmarks = if use_sim_data
    [[10.0, 0.0], [-4.0, -2.0], [-6.0, 5.0]]
else
    [[-1.230314, -0.606814], [0.797073, 0.889471], [-3.496525, 0.207874]]
end
landmarks_tensor = landmarks |> SEDL.hcatreduce |> x -> Flux.cat(x'; dims=3) |> device
@smart_assert size(landmarks_tensor) == (length(landmarks), 2, 1)

obs_model = if use_simple_obs_model
    let σs = use_sim_data ? (pos=0.1, θ=5°) : (pos=0.1, angle_2d=5°)
        state -> SEDL.gaussian_obs_model(state, σs)
    end
else
    state -> SEDL.landmark_obs_model(
        state, (; landmarks=landmarks_tensor, σ_bearing=10°, σ_range=5.0)
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
    rows = @showprogress 0.1 "posterior_metrics" map(1:n_repeats) do _
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
    is_quick_test && return nothing
    try
        task()
        alert("$task_name finished.")
    catch e
        alert("$task_name stopped due to exception: $(summary(e)).")
        rethrow()
    end
end

if use_sim_data
    plot_pf_posterior(motion_model, data_train, 1; title="true posterior") |> display
end
##-----------------------------------------------------------
# load or set up the NN models
function save_model_weights!()
    mm_weights = Flux.params(learned_motion_model)
    serialize(joinpath(save_dir, "mm_weights.bson"), mm_weights)
    if train_method == :VI
        guide_weights = Flux.params(guide)
        serialize(joinpath(save_dir, "guide_weights.bson"), guide_weights)
    end
end

function load_model_weights!()
    mm_weights = deserialize(joinpath(save_dir, "mm_weights.bson")) 
    Flux.loadparams!(learned_motion_model, mm_weights)
    if train_method == :VI
        guide_weights = deserialize(joinpath(save_dir, "guide_weights.bson"))
        Flux.loadparams!(guide, guide_weights)
    end
end

h_dim = 64
y_dim = sum(m -> size(m, 1), data_train.observations[1].val)

settings = (;
    is_quick_test,
    scenario=summary(sce),
    use_sim_data,
    train_method,
    h_dim,
    should_train_dynamics,
    use_simple_obs_model,
    n_train_ex,
)
save_dir = SEDL.data_dir(savename("train_models", settings; connector="-"))
if !load_trained && isdir(save_dir)
    @warn "removing old data at $save_dir..."
    rm(save_dir; recursive=true)
    @tagsave(joinpath(save_dir, "settings.bson"), @dict(settings))
end

logger = TBLogger(joinpath(save_dir, "tb_logs"))
@info """To view tensorboard logs, use the following command: 
```
tensorboard --samples_per_plugin "images=100" --logdir "$save_dir/tb_logs"
```"""

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

if load_trained
    @warn "Loading motion model weights from file..."
    load_model_weights!()
end

# adam = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.WeightDecay(1e-4), Flux.ADAM(1e-4))
adam = Flux.ADAM(1e-4)
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
                @info "testing" test_scores... log_step_increment = 0
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
    load_trained || save_model_weights!()
end
##-----------------------------------------------------------
# train the model using supervised learning (assuming having ground truth trajectories)
function supervised_callback(
    learned_motion_model::BatchedMotionModel; n_steps, test_every=100, plot_every=2_500
)
    prog = Progress(n_steps; showspeed=true)
    test_in_set, test_out_set = SEDL.input_output_from_trajectory(
        learned_motion_model.sketch, data_test.states, data_test.controls, data_test.times
    )
    test_input = BatchTuple(test_in_set)
    test_output = BatchTuple(test_out_set)
    compute_test_loss() =
        -SEDL.transition_logp(learned_motion_model.core, test_input, test_output) /
        test_input.batch_size

    early_stopping = SEDL.EarlyStopping(; max_iters_to_wait=5)

    function (r)
        Base.with_logger(logger) do
            @info "training" r.loss r.lr
            @info "statistics" r.time_stats... log_step_increment = 0
        end

        # Compute test log_obs and plot a few trajectories.
        if r.step % test_every == 1
            test_loss = compute_test_loss()
            Base.with_logger(logger) do
                @info("testing", test_loss, log_step_increment = 0)
            end
            should_stop = early_stopping(test_loss).should_stop
        else
            should_stop = false
        end
        if r.step % plot_every == 1
            test_metrics = posterior_metrics(learned_motion_model, data_test)

            Base.with_logger(logger) do
                @info("testing", test_metrics..., log_step_increment = 0)
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
        (; should_stop)
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
    load_trained || save_model_weights!()
end
##-----------------------------------------------------------
# train the guide using variational inference
function vi_callback(
    learned_motion_model::BatchedMotionModel; n_steps, trajs_per_ex=10, test_every=200
)
    prog = Progress(n_steps; showspeed=true)
    early_stopping = SEDL.EarlyStopping(; max_iters_to_wait=5)
    function (r)
        Base.with_logger(logger) do
            @info "training" r.elbo r.loss r.lr r.annealing
            @info "statistics" r.time_stats... log_step_increment = 0
        end

        test_x0_batch = repeat(data_test.states[1], trajs_per_ex)
        repeated_obs_seq = repeat.(data_test.observations, trajs_per_ex)
        repeated_control_seq = repeat.(data_test.controls, trajs_per_ex)
        # Compute test elbo and plot a few trajectories.
        if r.step % test_every == 1
            test_trajs, test_lp_guide, test_core_in, test_core_out = guide(
                test_x0_batch, repeated_obs_seq, repeated_control_seq, data_test.Δt
            )
            test_lp_dynamics = SEDL.transition_logp(
                learned_motion_model.core, test_core_in, test_core_out
            )
            test_lp_obs = SEDL.observation_logp(obs_model, test_trajs, repeated_obs_seq)
            test_elbo =
                (test_lp_dynamics + test_lp_obs - sum(test_lp_guide)) /
                (length(test_trajs) * test_x0_batch.batch_size)

            Base.with_logger(logger) do
                @info "testing" elbo = test_elbo log_step_increment = 0
            end

            test_scores = posterior_metrics(learned_motion_model, data_test)
            # should_stop = early_stopping(-test_elbo).should_stop
            should_stop = false

            Base.with_logger(logger) do
                @info "testing" test_scores... log_step_increment = 0
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
        else
            should_stop = false
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
        (; should_stop)
    end
end

if train_method == :VI
    load_trained || with_alert("VI training") do
        total_steps = 10_000
        n_steps = is_quick_test ? 3 : total_steps + 1

        n_repeat = 10
        vi_x0 = repeat(data_train.states[1], n_repeat)
        vi_obs_seq = repeat.(data_train.observations, n_repeat)
        vi_control_seq = repeat.(data_train.controls, n_repeat)

        @info "Training the guide..."
        train_result = @time SEDL.train_VI!(
            guide,
            learned_motion_model.core,
            obs_model,
            vi_x0,
            vi_obs_seq,
            vi_control_seq,
            data_train.Δt;
            optimizer=adam,
            n_steps,
            anneal_schedule=step -> linear(1e-3, 1.0)(min(1, 1.2step / n_steps)),
            callback=vi_callback(learned_motion_model; n_steps, trajs_per_ex=1),
            lr_schedule=let β = 40^2 / total_steps
                step -> 1e-4 / sqrt(β * step)
            end,
        )
        display(train_result)
    end
    # save the model weights
    load_trained || save_model_weights!()
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
    n_repeats = is_quick_test ? 2 : 5
    metrics = posterior_metrics(motion_model, data_test; n_repeats)
    open_loop = posterior_metrics(motion_model, data_test; n_repeats, obs_frames=1:1).RMSE
    (; name, metrics..., open_loop)
end

@info "Testing dynamics model performance on the test set"
perf = evaluate_model(learned_motion_model, data_test, string(train_method))
DataFrame([perf]) |> display
wsave(joinpath(save_dir, "logp_obs_table.bson"), @dict perf)
##-----------------------------------------------------------
1+2