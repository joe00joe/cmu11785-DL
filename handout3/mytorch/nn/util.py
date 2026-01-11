from mytorch import tensor
import numpy as np

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))


def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K) where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    
    # TODO: INSTRUCTIONS
    # Find the sorted indices based on number of time steps in each sample
    # Extract slices from each sample and properly order them for the construction of the packed tensor. __getitem__ you defined for Tensor class will come in handy
    # Use the tensor.cat function to create a single tensor from the re-ordered segements
    # Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.
    assert sequence is not None and type(sequence) == list
    assert len(sequence) > 0
    #sort

    sorted_indices, sorted_seqs = zip(*sorted(enumerate(sequence), key=lambda x: -x[1].shape[0]))
    # batch_sizes
    seq_lengths = np.array([seq.shape[0] for seq in sorted_seqs])
    max_len = seq_lengths.max()
    time_steps = np.arange(1, max_len + 1).reshape(-1, 1)
    batch_sizes = np.sum(seq_lengths >= time_steps, axis=1)
    #slice
    packed_data_list = []
    for t in range(max_len):
        cur_batch = batch_sizes[t]
        for i in range(cur_batch):
            packed_data_list.append(sorted_seqs[i][t].unsqueeze())

    #cat
    packed_data = tensor.cat(packed_data_list)
    return PackedSequence(packed_data, np.array(sorted_indices), batch_sizes)

     

def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    
    # TODO: INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list (assuming the tensors were sorted in a descending fashion based on number of timesteps)
    # Construct these individual tensors using tensor.cat
    # Re-arrange this list of tensor based on ps.sorted_indices
    assert type(ps.data) == tensor.Tensor and type(ps.batch_sizes) == np.ndarray
    assert len(ps.data) == sum(ps.batch_sizes)
    
    num_samples = ps.batch_sizes[0]
    max_len = len(ps.batch_sizes)
    slices_step = []
    start = 0
    for t in range(max_len):
        bsize = ps.batch_sizes[t]
        slices_step.append(ps.data[start:start+bsize])
        start += bsize
    seqs = []
    for i in range(num_samples):
        seq = []
        for t in range(max_len):
            if(i >= len(slices_step[t])):
                break
            seq.append(slices_step[t][i].unsqueeze())
        seqs.append( tensor.cat(seq))
    reorder_seqs = []
    sorted_indices =ps.sorted_indices.tolist()
    for i in range(len(sorted_indices)):
        j = sorted_indices.index(i)
        reorder_seqs.append(seqs[j])   
    return reorder_seqs

