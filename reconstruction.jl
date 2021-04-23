using LinearAlgebra
using HDF5
using IterativeSolvers

Dl = h5open("dictionaries.h5", "r") do file
    read(file, "Dl")
end

Dh = h5open("dictionaries.h5", "r") do file
    read(file, "Dh")
end

downsampled2_data = h5open("../sr_recon_code/train/sub001/recon/sub001_sc19_recon_downsampled_noint2.h5", "r") do file
    read(file,"recon")
end

groundtruth_data = h5open("../sr_recon_code/train/sub001/recon/sub001_sc19_recon.h5", "r") do file
    read(file,"recon")
end

time_slice = 41
