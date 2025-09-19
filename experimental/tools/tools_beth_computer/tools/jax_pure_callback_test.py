import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree

def generate_noisy_ellipsoid_sample_data(center:tuple[int,int,int]=(0,0,0),semi_axes:tuple[float,float,float]=(1.0,1.0,800/840),radius:float=1.0,theta_samples:int=20,seed:int=42,add_noise:bool=True):
    """
    modified from https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/
    """
    # create noise
    if add_noise:
        np.random.seed(seed)
        noise = np.random.normal(size=(theta_samples*theta_samples), loc=0, scale=1e-2)
    else:
        noise = 0
        #noise = np.zeros((theta_samples*theta_samples,))

    # define u,v space which equate to theta and phi in spherical coordinates
    u = np.linspace(0.,np.pi*2,theta_samples)
    v = np.linspace(0.,np.pi, theta_samples)
    u,v = np.meshgrid(u,v,sparse=True)
    a = semi_axes[0] #1.0
    b = semi_axes[1] #0.5,
    c = semi_axes[2] #800/840

    # calculate cartesian coordinates from spherical
    x = a*np.cos(u)*np.sin(v)*radius
    y = b*np.cos(v)*radius
    z = c*np.sin(u)*np.sin(v)*radius

    x = x.flatten() + noise
    y = np.repeat(y.flatten(),theta_samples) + noise
    z = z.flatten() + noise

    x = x+center[0]
    y = y+center[1]
    z = z+center[2]

    return np.column_stack((x,y,z))

############################################################

# 1. Define the custom non-JAX (host-side) function
def numpy_sin_host(x):
    """A pure function that uses numpy (not jax.numpy)."""
    # x will be a concrete NumPy array on the host CPU
    return np.sin(x).astype(x.dtype)

def my_host_func(params,points_3D):
    """"""
    center = params[:3]
    coeff = params[3:]
    radius = np.sqrt(np.sum((points_3D - center)**2,axis=1)).mean()
    ellipsoid_points = generate_noisy_ellipsoid_sample_data(center,semi_axes=coeff,radius=radius)
    tree = KDTree(ellipsoid_points)
    distances,indices = tree.query(points_3D,k=1)
    return distances #radius

# 2. Define the JAX wrapper function
@jax.jit
def jax_function_with_callback(x_jax):
    # Specify the expected shape and dtype of the output
    result_shape_dtype = jax.ShapeDtypeStruct(x_jax.shape, x_jax.dtype)
    
    # Call the host function using jax.pure_callback
    # The arguments (x_jax) are passed to the callback
    host_result = jax.pure_callback(
        numpy_sin_host,
        result_shape_dtype,
        x_jax,
        vmap_method='sequential' # explicitly handle vmap behavior
    )
    
    # You can now use the result in subsequent JAX computations
    return host_result * 2.0

@jax.jit
def my_jax_function_with_callback(params_jax,points_3d_jax):
    """
    """
    #shape = ()
    shape = points_3D[:,0].shape
    result_shape_dtype = jax.ShapeDtypeStruct(shape,params_jax.dtype)
    my_result = jax.pure_callback(
        my_host_func,
        result_shape_dtype,
        params_jax,
        points_3d_jax,
        vmap_method='sequential'
    )
    return jnp.var(my_result)

# 3. Test the function
x = jnp.arange(5.0)
y = jax_function_with_callback(x)

print("JAX input array:", x)
print("Result of JIT-compiled function with pure_callback:", y)
print("Type of the result:", type(y))

# my test data
params = np.array([0.,7.5,0.,1.,1.,1.])
points_3D = np.array([[0,1,0],[1,0,1],[0,0,1]])
print(points_3D.shape)

result = my_jax_function_with_callback(params,points_3D)
print(f"Result of JIT-compiled function with pure_callback:{np.sqrt(result)}\nResult type: {type(result)}\n")