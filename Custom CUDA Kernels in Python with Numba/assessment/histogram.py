# Add your solution here
@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    '''Increment bin counts in histogram_out, given histogram range [xmin, xmax).'''
    start = cuda.grid(1) # idx
    stride = cuda.gridsize(1)
    
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins
    
    for i in range(start, x.shape[0], stride):
        bin_number = np.int32((x[i] - xmin) / bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            # only increment if in range
            cuda.atomic.add(histogram_out, bin_number, 1)  # Safely add 1 to offset 0 in global_counter array