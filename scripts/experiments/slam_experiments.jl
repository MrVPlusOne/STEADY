include("experiment_common.jl")
using DataFrames

perfs = let script_args = (;
    validation_metric=:RMSE,
    # n_train_ex=256,
    gpu_id=Main.GPU_ID, # set this in the REPL before running the script
    # use_fixed_variance=true,
    # use_simple_obs_model=true,
    train_method=:EM_SLAM,
)

    train_multiple_times(script_args, 1).test_performance
end

display(DataFrame([map(SEDL.to_measurement, perfs)]))
alert("SLAM experiments finished. (GPU ID: $(Main.GPU_ID))")