is_quick_test=true

ENV["GKSwstype"]="nul"
@warn "Plot display turned off via 'GKSwstype'."

include("../test_vi_new.jl")