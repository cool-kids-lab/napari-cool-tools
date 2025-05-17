import torch
import torch.nn.functional as F

# nn Model utility functions

def _squash(input_tensor):
    norm = torch.norm(input_tensor,dim=-1,keepdim=True) #tf.norm(input_tensor, axis=-1, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


# code modified from https://github.com/lalonderodney/SegCaps
def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                    num_routing,device):
    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = votes.permute(votes_t_shape) #tf.transpose(votes, votes_t_shape) # replace tf.transpose
    _, _, _, height, width, caps = votes_trans.shape

    #activations = torch.empty(num_routing,dtype=torch.float32)
    activations = []
    logits = torch.zeros(logit_shape.tolist()).to(device) # can we optimize out the tolist()
    #print(f"logits:{type(logits)}\n")
    
    for i in range(num_routing):
        
        route = F.softmax(logits,dim=-1).to(device)
        #print(f"route type: {route.dtype}, votes_trans type: {votes_trans.dtype}\n")
        preactivate_unrolled = route * votes_trans
        preact_trans = preactivate_unrolled.permute(r_t_shape) #tf.transpose(preactivate_unrolled, r_t_shape) #tf.transpose
        preactivate = preact_trans.sum(dim=1) + biases #tf.reduce_sum(preact_trans, axis=1) + biases #tf.reduce_sum
        activation = _squash(preactivate) # squash
        #activations[i] = activation #activations.write(i, activation)
        activations.append(activation)
        act_3d = activation.unsqueeze(dim=1) #K.expand_dims(activation, 1)
        tile_shape = torch.ones(num_dims,dtype=torch.int32).tolist() #np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = act_3d.tile(tile_shape) #tf.tile(act_3d, tile_shape)
        distances = torch.sum((votes*act_replicated),dim=-1) # tf.reduce_sum(votes * act_replicated, axis=-1)
        logits += distances

    return activations[num_routing - 1].to(torch.float32) #K.cast(activations.read(num_routing - 1), dtype='float32') # K.cast