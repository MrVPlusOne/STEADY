include("experiments/experiment_common.jl")
include("data_from_source.jl")

scenario = SEDL.RealCarScenario("alpha_truck")
data_source = if scenario.data_name == "alpha_truck"
    # lagacy format
    SeparateData(;
        train_data_dir="datasets/alpha_truck/train",
        valid_data_dir="datasets/alpha_truck/valid",
        test_data_dir="datasets/alpha_truck/test",
    )
else
    MixedData(;
        data_dir="datasets/$(scenario.data_name)",
        test_data_ratio=0.25,
        valid_data_ratio=0.25,
    )
end

data_cache_path = let
    cache_name = savename(
        (;
            scenario=string(scenario),
            source=string(data_source),
            use_simple_obs_model,
            σ_bearing,
        ),
        "serial";
        connector="-",
    )
    SEDL.data_dir("data_cache", cache_name)
end
