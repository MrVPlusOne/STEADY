using SEDL
my_include = include # to avoid mess up the VSCode linter

global script_args = (;
    is_quick_test=false,
    gpu_id=6,
    use_fixed_variance=false,
    use_simple_obs_model=true,
    train_method=:VI,
    max_trian_steps=10_000,
)
my_include("../train_models.jl")
DataFrame([Main.perf]) |> display