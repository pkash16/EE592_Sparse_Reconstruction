using LinearAlgebra
using HDF5
using ImageView
using MIRT
using Plots
using IterativeSolvers
using LinearOperators

include("helpers.jl")


train_dir = "../sr_recon_code/train/"
c = h5open("../sr_recon_code/train/sub001/recon/sub001_sc19_recon_downsampled_noint2.h5", "r") do file
    global downsampled2_data = read(file,"recon")
end

c = h5open("../sr_recon_code/train/sub001/recon/sub001_sc19_recon.h5", "r") do file
    global groundtruth_data = read(file,"recon")
end

#let's generate the dictionary based off one file as a test.
GTwh = Int32(12);
LRwh = Int32(6);
time_slices = 40:50 #take a small subset of time slices

num_patches = Int32(floor(prod(size(groundtruth_data[:,:,time_slices[1]])) / (GTwh * GTwh))) * length(time_slices)
dict_high = zeros(GTwh*GTwh, num_patches)
dict_low = zeros(LRwh * LRwh, num_patches)
dict_high_idx = 1
dict_low_idx = 1


for time in time_slices
    sample_gt = groundtruth_data[:,:,time]
    size_samp_gt = size(sample_gt)
    num_patches_gt = prod(size_samp_gt) / (GTwh*GTwh);
    print(num_patches_gt)

    for i = 1:floor(size_samp_gt[1]/GTwh)
        for j = 1:floor(size_samp_gt[2]/GTwh)
            dict_high[:, dict_high_idx] = reshape(sample_gt[ Int32(GTwh*(i-1) + 1) : Int32(GTwh*i), Int32(GTwh*(j-1) + 1):Int32(GTwh*j)], (GTwh*GTwh), 1)
            dict_high_idx = dict_high_idx + 1
        end
    end

    sample_lr = downsampled2_data[:,:,time]
    size_samp_lr = size(sample_lr)
    num_patches_lr = prod(size_samp_lr) / (LRwh * LRwh)
    for i = 1:floor(size_samp_lr[1]/LRwh)
        for j = 1:floor(size_samp_lr[2]/LRwh)
            dict_low[:, dict_low_idx] = reshape(sample_lr[ Int32(LRwh*(i-1) + 1) : Int32(LRwh*i), Int32(LRwh*(j-1) + 1):Int32(LRwh*j)], (LRwh*LRwh), 1)
            dict_low_idx = dict_low_idx + 1
        end
    end
end

#Let's try this again, but this time construct high AND low resolution patches.
LOW_PATCH_SIZE = size(dict_low, 1)
HIGH_PATCH_SIZE = size(dict_high, 1)
NUM_PATCHES = 20;

D = randn(LOW_PATCH_SIZE + HIGH_PATCH_SIZE, NUM_PATCHES)
X̃ = [dict_low / sqrt(LOW_PATCH_SIZE); dict_high / sqrt(HIGH_PATCH_SIZE)]
Z = randn(NUM_PATCHES, size(X̃, 2))


for iter=1:50
    Z = updateZ(Z,D,X̃, 10)
    D = updateD(D,Z,X̃, 1000)
end

Dl = D[1:LOW_PATCH_SIZE, :] .* sqrt(LOW_PATCH_SIZE)
Dh = D[LOW_PATCH_SIZE+1:end, :] .* sqrt(HIGH_PATCH_SIZE)

#export to data files to read in "reconstruction.jl"
h5open("dictionaries.h5", "w") do file
    write(file, "Dl", Dl)
    write(file, "Dh", Dh)
end

#now that we have dictionary.... let's try the sparse encoding reconstruction!
#This will be in a different file... reconstruction.jl?
