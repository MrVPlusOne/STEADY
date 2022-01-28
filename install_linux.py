# install the dependences for the project (linux version)
# a python 3 script

from shutil import which
import subprocess
from pathlib import Path
import os
from os.path import exists, expanduser
import sys

dep_path = Path(expanduser("~/SEDL-dependences"))
dep_path.mkdir(parents=True, exist_ok=True)
print(f"Dependencies will be installed in the folder '{dep_path}'")
os.chdir(dep_path)

conda_path = Path(expanduser("~/miniconda3"))
path_list = []

# first, install julia
if which("julia") is None:
    print("Installing Julia...")
    subprocess.call(["wget", "https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.1-linux-x86_64.tar.gz", "-O", "julia-install.tar.gz"])
    subprocess.call(["tar", "-xf", "julia-install.tar.gz"])
    subprocess.call(["rm", "julia-install.tar.gz"])
    path_list.append(f"{os.getcwd()}/julia-1.7.1/bin")

julia_startup_path = Path(expanduser("~/.julia/config/startup.jl"))
if not exists(julia_startup_path):
    print("Configuring Julia...")
    startup_code = """
try
    using Revise
    @info("Revise in use.")
    using Alert
    using AlertPushover: pushover_alert!
    pushover_alert!(
        token = "atvgvzskpz4fsudfq8wzwms8jc8sqy",
        user = "ugpawc57ura8d8f4u3quia8dtud8oe",
    )
    @info("AlertPushover in use.")
catch e
    @warn e.msg
end
"""
    # create the directory if it does not exist
    julia_startup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(julia_startup_path, "w") as f:
        f.write(startup_code)

if which("conda") is None:
    print("Installing conda...")
    subprocess.call(["wget", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh", "-O", "conda-install.sh"])
    subprocess.call(["bash", "conda-install.sh", "-b"])
    subprocess.call(["rm", "conda-install.sh"])
    path_list.append(f"{conda_path}/bin")

if which("tensorboard") is None:
    # create a new environment named "SEDL"
    print("Installing Tensorboard into a environment called `sedl` ...")
    subprocess.call([conda_path / "bin/conda", "create", "tensorboard", "-n", "sedl"])
    path_list.append(f"{conda_path}/sedl/bin")


print("Installation finished.")
if len(path_list) > 0:
    print("Now, add the following to your .bashrc:\n```")
    for np in path_list:
        print(f"export PATH=$PATH:{np}")
    print("```")