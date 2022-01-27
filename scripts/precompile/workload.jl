ENV["GKSwstype"] = "nul"
@warn "Plot display turned off via 'GKSwstype'."

global script_args = (;
    is_quick_test=true,
    gpu_id=1,
    use_sim_data=false,
    use_simple_obs_model=true,
    train_method=:EM,
)

my_include = include # to avoid mess up the VSCode linter
my_include("../train_models.jl")