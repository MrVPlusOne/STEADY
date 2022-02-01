using SEDL
my_include = include # to avoid mess up the VSCode linter

global script_args = (;
    is_quick_test=false,
    gpu_id=1,
    use_fixed_variance=false,
    use_simple_obs_model=true,
    train_method=:VI,
    max_train_steps=10_000,
)
my_include("../train_models.jl")
exp_info = (architecture="Simplified VI", scenario="Hovercraft", obs_model="Gaussian with velocity")
println("Experiment setting: $exp_info")
DataFrame([Main.perf]) |> display