##-----------------------------------------------------------
# load trained model
using Flux
include("experiment_common.jl")

steady_model, fit_hand_model = map([:EM, :FitHand]) do train_method
    args = (
        load_trained=true,
        gpu_id=Main.GPU_ID,
        scenario=SEDL.RealCarScenario("alpha_truck"),
        exp_group="comparisons",
        train_method,
    )
    @eval(Main, script_args = $args)
    dynamic_include("../train_models.jl")
    Main.learned_motion_model
end
##-----------------------------------------------------------
# plot local velocities
for motion_model in [steady_model, fit_hand_model]
    data = Main.data_test
    sample_id = 3

    pf_trajs, core_in, core_out = SEDL.sample_posterior_pf(
        motion_model,
        Main.logpdf_obs,
        data,
        sample_id;
        obs_frames=1:10:length(data.times),
        record_io=true,
    )
    core_in_truth, core_out_truth = SEDL.input_output_from_trajectory(
        motion_model.sketch,
        getindex.(data.states, sample_id),
        getindex.(data.controls, sample_id),
        data.times,
    )
    series = map(core_in) do b
        BatchTuple(b.tconf, b.batch_size, (loc_v=b.val.loc_v,))
    end
    plot_batched_series(
        data.times[1:(end - 1)],
        series;
        truth=core_in_truth,
        component_names=Dict(:loc_v => ["longitudinal velocity", "lateral velocity"]),
        size=(440, 300),
        yrange=(-0.5, 2.5),
    ) |> display
end
##-----------------------------------------------------------