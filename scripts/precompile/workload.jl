
global script_args = (;
    is_quick_test=true,
    gpu_id=1,
    use_simple_obs_model=true,
    train_method=:EM,
    n_particles=1000,
)

my_include = include # to avoid mess up the VSCode linter
my_include("../turn_off_displays.jl")
my_include("../train_models.jl")