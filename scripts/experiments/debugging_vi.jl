using SEDL
my_include = include # to avoid mess up the VSCode linter

global script_args = (;
    gpu_id=Main.GPU_ID,
    # scenario=SEDL.HovercraftScenario(),
    use_fixed_variance=false,
    # use_simple_obs_model=true,
    train_method=:VI,
    # n_train_ex=128,
)
my_include("../train_models.jl")

DataFrame([Main.perf]) |> display