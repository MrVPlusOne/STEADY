using SEDL
my_include = include # to avoid mess up the VSCode linter

global script_args = (;
    gpu_id=6,
    scenario=SEDL.HovercraftScenario(),
    use_fixed_variance=false,
    use_simple_obs_model=true,
    train_method=:VI,
    n_train_ex=128,
    max_train_steps=10_000,
)
my_include("../train_models.jl")
exp_info = (
    architecture="Simplified VI",
    scenario="Hovercraft",
    obs_model="Gaussian with velocity",
    n_train_ex=128,
)
println("Experiment setting: $exp_info")
DataFrame([Main.perf]) |> display