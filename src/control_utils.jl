module SymbolMaps
export SymbolMap, submap
export integral!, derivative!

import Base./
using Base: @kwdef

const symbol_path_cache = Dict{Tuple{Symbol,Symbol},Symbol}()
function symbol_path(s1::Symbol, s2::Symbol)::Symbol
    get!(symbol_path_cache, (s1, s2)) do
        Symbol(s1, :/, s2)
    end
end

"""
Append the symbol `s2` to symbol `s1`.
"""
@inline function /(s1::Symbol, s2::Symbol)
    symbol_path(s1, s2)
end

@kwdef(struct SymbolMap
    base_path::Symbol = :/
    data::Dict{Symbol,Any} = Dict{Symbol,Any}()
end)

Base.getindex(m::SymbolMap, path::Symbol) = m.data[m.base_path / path]
Base.setindex!(m::SymbolMap, v, path::Symbol) = m.data[m.base_path / path] = v
Base.get!(m::SymbolMap, path::Symbol, default) = get!(m.data, m.base_path / path, default)
Base.get(m::SymbolMap, path::Symbol, default) = get(m.data, m.base_path / path, default)

"""
Creat a restricted view of the orginal SymbolMap. 
"""
function submap(m::SymbolMap, path::Symbol)::SymbolMap
    SymbolMap(m.base_path / path, m.data)
end

function integral!(m::SymbolMap, path::Symbol, t::T, e::E)::E where {T,E}
    r = get(m, path, nothing)
    if r === nothing
        (m[path] = (t, zero(E)))[2]
    else
        (t0, v0) = r::Tuple{T,E}
        @assert(
            t > t0,
            "`integral!` with the path `$path` shouldn't be called multiple 
    times in a single time step. (t = $t)"
        )
        Δt = t - t0
        (m[path] = (t, v0 + e * Δt))[2]
    end
end

function derivative!(m::SymbolMap, path::Symbol, t::T, e::E)::E where {T,E}
    r = get(m, path, nothing)
    if r === nothing
        (m[path] = (t, zero(E)))[2]
    else
        (t0, v0) = r::Tuple{T,E}
        @assert(
            t > t0,
            "`derivative!` with the path `$path` shouldn't be called multiple 
    times in a single time step. (t = $t)"
        )
        Δt = t - t0
        (m[path] = (t, v0 + e / Δt))[2]
    end
end

end # module SymbolMaps